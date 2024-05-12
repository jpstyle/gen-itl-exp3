using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.SideChannels;

public class TeacherAgent : DialogueAgent
{
    // Keep references to prefabs/materials for initialization at each episode reset
    [SerializeField]
    private List<GameObject> cabinTypes;
    [SerializeField]
    private List<GameObject> loadTypes;
    [SerializeField]
    private List<GameObject> frontChassisTypes;
    [SerializeField]
    private List<GameObject> centerChassisTypes;
    [SerializeField]
    private List<GameObject> backChassisTypes;
    [SerializeField]
    private List<GameObject> fenderFrontLeftTypes;
    [SerializeField]
    private List<GameObject> fenderFrontRightTypes;
    [SerializeField]
    private List<GameObject> fenderBackLeftTypes;
    [SerializeField]
    private List<GameObject> fenderBackRightTypes;
    [SerializeField]
    private List<GameObject> wheelTypes;
    [SerializeField]
    private List<GameObject> boltTypes;
    [SerializeField]
    private List<Material> colors;

    // Store workplace partition info for initializing part poses
    private readonly List<Vector3> _partPartitionPositions = new()
    {
        new Vector3(-0.24f, 0.76f, 0.24f),
        new Vector3(0.24f, 0.76f, 0.24f),
        new Vector3(-0.36f, 0.76f, 0.48f),
        new Vector3(-0.12f, 0.76f, 0.48f),
        new Vector3(0.12f, 0.76f, 0.48f),
        new Vector3(0.36f, 0.76f, 0.48f),
        new Vector3(-0.42f, 0.88f, 0.74f),
        new Vector3(-0.14f, 0.88f, 0.74f),
        new Vector3(0.14f, 0.88f, 0.74f),
        new Vector3(0.42f, 0.88f, 0.74f)
    };
    // Store workplace partition info for main work area
    private readonly Vector3 _mainPartitionPosition = new(0f, 0.76f, 0.24f);
    
    protected override void Awake()
    {
        // Register Python-Agent string communication side channel
        // (Note: This works because we will have only one instance of the agent
        // in the scene ever, but ideally we would want 1 channel per instance,
        // with UUIDs generated on the fly each time an instance is created...)
        channelUuid = "da85d4e0-1b60-4c8a-877d-03af30c446f2";
        backendMsgChannel = new MessageSideChannel(channelUuid, this);
        SideChannelManager.RegisterSideChannel(backendMsgChannel);

        Academy.Instance.OnEnvironmentReset += EnvironmentReset;
    }

    private void EnvironmentReset()
    {
        // Fetch environment parameters received from python backend
        var envParams = Academy.Instance.EnvironmentParameters;

        // Setup episode only if any environment parameters are provided
        if (envParams.Keys().Count == 0) return;

        // Destroy any existing objects with EnvEntity component
        foreach (var ent in FindObjectsByType<EnvEntity>(FindObjectsSortMode.None))
        {
            ent.enabled = false;
                // Disable EnvEntity component; needed because Destroy calls are delayed
            Destroy(ent.gameObject);
        }
        // Disable all existing RigidBody components; also because Destroy calls are delayed
        foreach (var rb in FindObjectsByType<Rigidbody>(FindObjectsSortMode.None))
            rb.detectCollisions = false;

        // Sample configs of parts that make up a valid truck
        var cabinType = cabinTypes[
            (int) envParams.GetWithDefault("cabin_type", Random.Range(0, cabinTypes.Count))
        ];
        var loadType = loadTypes[
            (int) envParams.GetWithDefault("load_type", Random.Range(0, loadTypes.Count))
        ];
        var centerChassisType = centerChassisTypes[Random.Range(0, centerChassisTypes.Count)];
        // We consider only three combinations of fenders & wheels for sampling; front-normal
        // + back-normal, front-large + back-large, front-normal + back-double. Initialize
        // wheels with appropriate and size and number.
        var fenderWheelCombos = new List<(int, int, int, int, int, int, int)>
        {
            // Fender-FL, Fender-FR, Fender-BL, Fender-BR, Wheel, NumWheels, NumBolts
            (0, 0, 0, 0, 0, 4, 9),
            (1, 1, 1, 1, 1, 4, 9),
            (0, 0, 2, 2, 0, 6, 11),
        };
        var fenderWheelComboType = fenderWheelCombos[Random.Range(0, 3)];

        // Sample part colorings
        var partColors = new Dictionary<string, Material>
        {
            ["cabin"] = colors[Random.Range(0, colors.Count)],
            ["chassis_center"] = colors[Random.Range(0, colors.Count)],
            ["fender_front_left"] = colors[Random.Range(0, colors.Count)],
            ["fender_front_right"] = colors[Random.Range(0, colors.Count)],
            ["fender_back_left"] = colors[Random.Range(0, colors.Count)],
            ["fender_back_right"] = colors[Random.Range(0, colors.Count)],
        };

        // Certain combinations of load & center chassis collide and should be rejected
        var combosToReject = new List<(string, string)>
        {
            ("load_dumper", "chassis_center_spares"),
            ("load_dumper", "chassis_center_staircase_4seats"),
            ("load_dumper", "chassis_center_staircase_oshkosh"),
            ("load_rocketLauncher", "chassis_center_spares")
        };
        while (true)        // Rejection sampling
        {
            if (!combosToReject.Contains((loadType.name, centerChassisType.name)))
                break;

            loadType = loadTypes[(int) envParams.GetWithDefault("load_type", 0f)];
            centerChassisType = centerChassisTypes[Random.Range(0, centerChassisTypes.Count)];
        }

        // Create appropriate number of 'workspace partitions'; square areas that take up
        // space on the tabletop (corresponding gameObjects don't need to be Planes though,
        // we only need reference coordinates)
        var addDistractors = envParams.GetWithDefault("_add_distractors", 0f) == 0f;
        var numPartitions = addDistractors ? 10 : 6;
        var sampledPartitionLocations = Enumerable.Range(0, _partPartitionPositions.Count).ToList();
        Shuffle(sampledPartitionLocations);

        var partitions = new List<GameObject>();
        for (var i = 0; i < numPartitions; i++)
        {
            var newPartition = new GameObject($"partition_{i}")
            {
                transform =
                {
                    position = _partPartitionPositions[sampledPartitionLocations[i]]
                }
            };
            partitions.Add(newPartition);
        }

        // Instantiate sampled parts on sampled partitions on tabletop
        InstantiateCabin(cabinType, partitions[0], partColors["cabin"]);
        InstantiateLoad(loadType, partitions[1]);
        InstantiateChassis(
            new List<GameObject>{ frontChassisTypes[0], centerChassisType, backChassisTypes[0] },
            partitions[2], partColors["chassis_center"]
        );
        InstantiateFenders(
            new List<GameObject>
            {
                fenderFrontLeftTypes[fenderWheelComboType.Item1],
                fenderFrontRightTypes[fenderWheelComboType.Item2],
                fenderBackLeftTypes[fenderWheelComboType.Item3],
                fenderBackRightTypes[fenderWheelComboType.Item4]
            },
            partitions[3],
            new List<Material>
            {
                partColors["fender_front_left"],
                partColors["fender_front_right"],
                partColors["fender_back_left"],
                partColors["fender_back_right"]
            }
        );
        InstantiateWheels(
            wheelTypes[fenderWheelComboType.Item5],
            fenderWheelComboType.Item6,
            partitions[4]
        );
        InstantiateBolts(boltTypes[0], fenderWheelComboType.Item7, partitions[5]);

        // Instantiate selected truck type
        // var truck = Instantiate(truckType);
        // truck.name = "truck";

        // Replace truck part prefabs
        // var sampledPartsWithHandles = new List<(string, GameObject)>
        // {
        //     ("cabin", cabinType), ("load", loadType), ("chassis_center", centerChassisType)
        // };
        // foreach (var (partType, sampledPart) in sampledPartsWithHandles)
        // {
            // var partSlotTf = truck.transform.Find(partType);
            // foreach (Transform child in partSlotTf)
            // {
            //     var replacedGObj = child.gameObject;
            //     replacedGObj.SetActive(false);         // Needed for UpdateClosestChildren below
            //     Destroy(replacedGObj);
            // }
            // var newPart = Instantiate(sampledPart, partSlotTf);
            // newPart.name = sampledPart.name;        // Not like 'necessary' but just coz
        // }
        // Need to update the closest children after the replacements
        // truck.GetComponent<EnvEntity>().UpdateClosestChildren();

        // Color parts if applicable
        // foreach (Transform partSlotTf in truck.transform)
        // {
        //     var partType = partSlotTf.gameObject.name;
        //     var matchingColorGroups =
        //         partColorGroups.Where(kv => partType.StartsWith(kv.Key)).ToList();
        //     if (matchingColorGroups.Count == 0) continue;
        //
        //     foreach (var mesh in partSlotTf.gameObject.GetComponentsInChildren<MeshRenderer>())
        //     {
        //         // Change material only if current one is Default one
        //         if (mesh.material.name.StartsWith("Default"))
        //             mesh.material = matchingColorGroups[0].Value;
        //     }
        // }

        // Random initialization of truck pose
        // truck.transform.position = new Vector3(
        //     Random.Range(-0.25f, 0.25f), 0.85f, Random.Range(0.3f, 0.35f)
        // );
        // truck.transform.eulerAngles = new Vector3(
        //     0f, Random.Range(0f, 359.9f), 0f
        // );

        // Fast-forward physics simulation until truck rests on the desktop
        Physics.simulationMode = SimulationMode.Script;
        for (var i=0; i<2000; i++)
        {
            Physics.Simulate(Time.fixedDeltaTime);
        }
        Physics.simulationMode = SimulationMode.FixedUpdate;

        // Currently stored annotation info is now obsolete
        EnvEntity.annotationStorage.annotationsUpToDate = false;
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (actionBuffers.DiscreteActions[0] == 1)
        {
            // 'Utter' action
            StartCoroutine(Utter());
        }
    }

    public override void Heuristic(in ActionBuffers actionBuffers)
    {
        // Update annotation whenever needed
        if (!EnvEntity.annotationStorage.annotationsUpToDate)
            StartCoroutine(CaptureAnnotations());

        // 'Utter' any outgoing messages
        if (outgoingMsgBuffer.Count > 0)
        {
            var discreteActionBuffers = actionBuffers.DiscreteActions;
            discreteActionBuffers[0] = 1;      // 'Utter'
        }
    }

    // Helper methods for initializing sampled parts on sampled locations
    private static void InstantiateCabin(
        GameObject cabinPrefab, GameObject partition, Material cabinColor
    )
    {
        // Instantiate provided cabin type with generalized name
        var cabin = Instantiate(cabinPrefab, partition.transform);
        cabin.name = "cabin";

        // Apply sampled color
        foreach (var mesh in cabin.GetComponentsInChildren<MeshRenderer>())
            if (mesh.material.name.StartsWith("Default")) mesh.material = cabinColor;

        // Add RigidBody component for physical interaction with environment
        cabin.AddComponent<Rigidbody>();
        
        // Define position w.r.t. partition coordinate
        cabin.transform.localPosition = new Vector3(0f, 0.06f, 0f);

        // Apply random rotations to the parent partition object
        var rotY = Random.Range(0f, 359.9f);
        partition.transform.eulerAngles = new Vector3(0f, rotY, 0f);
        
        // Detach instantiated part from parent
        cabin.transform.parent = null;
    }
    private static void InstantiateLoad(GameObject loadPrefab, GameObject partition)
    {
        // Instantiate provided load type with generalized name
        var load = Instantiate(loadPrefab, partition.transform);
        load.name = "load";

        // Add RigidBody component for physical interaction with environment
        load.AddComponent<Rigidbody>();
        
        // Define position w.r.t. partition coordinate
        load.transform.localPosition = new Vector3(0.04f, 0.02f, 0f);

        // Apply random rotations to the parent partition object
        var rotY = 180f * Random.Range(0, 2) + Random.Range(-10f, 10f);
        partition.transform.eulerAngles = new Vector3(0f, rotY, 0f);
        
        // Detach instantiated part from parent
        load.transform.parent = null;
    }
    private static void InstantiateChassis(
        List<GameObject> chassisPrefabs, GameObject partition, Material centerColor
    )
    {
        // Instantiate provided chassis types with generalized names
        var chassisFront = Instantiate(chassisPrefabs[0], partition.transform);
        var chassisCenter = Instantiate(chassisPrefabs[1], partition.transform);
        var chassisBack = Instantiate(chassisPrefabs[2], partition.transform);
        chassisFront.name = "chassis_front";
        chassisCenter.name = "chassis_center";
        chassisBack.name = "chassis_back";

        // Apply sampled color to center chassis
        foreach (var mesh in chassisCenter.GetComponentsInChildren<MeshRenderer>())
            if (mesh.material.name.StartsWith("Default")) mesh.material = centerColor;

        // Add RigidBody component for physical interaction with environment
        chassisFront.AddComponent<Rigidbody>();
        chassisCenter.AddComponent<Rigidbody>();
        chassisBack.AddComponent<Rigidbody>();
        
        // Define positions w.r.t. partition coordinate
        chassisFront.transform.localPosition = new Vector3(0.018f, 0.02f, -0.05f);
        chassisCenter.transform.localPosition = new Vector3(-0.075f, 0.02f, -0.05f);
        chassisBack.transform.localPosition = new Vector3(0f, 0.02f, 0.05f);

        // Apply random rotations to the parent partition object
        var rotY = 90f * Random.Range(0, 4) + Random.Range(-15f, 15f);
        partition.transform.eulerAngles = new Vector3(0f, rotY, 0f);
        
        // Detach instantiated parts from parent
        chassisFront.transform.parent = null;
        chassisCenter.transform.parent = null;
        chassisBack.transform.parent = null;
    }
    private static void InstantiateFenders(
        List<GameObject> fenderPrefabs, GameObject partition, List<Material> fenderColors
    )
    {
        // Instantiate provided fender types with generalized names
        var fenderFrontLeft = Instantiate(fenderPrefabs[0], partition.transform);
        var fenderFrontRight = Instantiate(fenderPrefabs[1], partition.transform);
        var fenderBackLeft = Instantiate(fenderPrefabs[2], partition.transform);
        var fenderBackRight = Instantiate(fenderPrefabs[3], partition.transform);
        fenderFrontLeft.name = "fender_front_left";
        fenderFrontRight.name = "fender_front_right";
        fenderBackLeft.name = "fender_back_left";
        fenderBackRight.name = "fender_back_right";

        // Apply sampled color to each fender
        foreach (var mesh in fenderFrontLeft.GetComponentsInChildren<MeshRenderer>())
            if (mesh.material.name.StartsWith("Default")) mesh.material = fenderColors[0];
        foreach (var mesh in fenderFrontRight.GetComponentsInChildren<MeshRenderer>())
            if (mesh.material.name.StartsWith("Default")) mesh.material = fenderColors[1];
        foreach (var mesh in fenderBackLeft.GetComponentsInChildren<MeshRenderer>())
            if (mesh.material.name.StartsWith("Default")) mesh.material = fenderColors[2];
        foreach (var mesh in fenderBackRight.GetComponentsInChildren<MeshRenderer>())
            if (mesh.material.name.StartsWith("Default")) mesh.material = fenderColors[3];

        // Add RigidBody component for physical interaction with environment
        fenderFrontLeft.AddComponent<Rigidbody>();
        fenderFrontRight.AddComponent<Rigidbody>();
        fenderBackLeft.AddComponent<Rigidbody>();
        fenderBackRight.AddComponent<Rigidbody>();
        
        // Define positions & rotations w.r.t. partition coordinate
        var zPositions = new List<float> {-0.10f, -0.04f, 0.02f, 0.08f};
        var randomIndices = Enumerable.Range(0, 4).ToList();
        Shuffle(randomIndices);
        fenderFrontLeft.transform.localPosition = new Vector3(
            Random.Range(-0.03f, 0.03f), 0.02f, zPositions[randomIndices[0]]
        );
        fenderFrontRight.transform.localPosition = new Vector3(
            Random.Range(-0.03f, 0.03f), 0.02f, zPositions[randomIndices[1]]
        );
        fenderBackLeft.transform.localPosition = new Vector3(
            Random.Range(-0.03f, 0.03f), 0.02f, zPositions[randomIndices[2]]
        );
        fenderBackRight.transform.localPosition = new Vector3(
            Random.Range(-0.03f, 0.03f), 0.02f, zPositions[randomIndices[3]]
        );
        fenderFrontLeft.transform.eulerAngles = new Vector3(
            (Random.Range(0, 2) * 2 - 1) * 90f, 0f, 0f
        );
        fenderFrontRight.transform.eulerAngles = new Vector3(
            (Random.Range(0, 2) * 2 - 1) * 90f, 0f, 0f
        );
        fenderBackLeft.transform.eulerAngles = new Vector3(
            (Random.Range(0, 2) * 2 - 1) * 90f, 0f, 0f
        );
        fenderBackRight.transform.eulerAngles = new Vector3(
            (Random.Range(0, 2) * 2 - 1) * 90f, 0f, 0f
        );

        // Apply random rotations to the parent partition object
        var rotY = 90f * Random.Range(0, 4) + Random.Range(-5f, 5f);
        partition.transform.eulerAngles = new Vector3(0f, rotY, 0f);
        
        // Detach instantiated parts from parent
        fenderFrontLeft.transform.parent = null;
        fenderFrontRight.transform.parent = null;
        fenderBackLeft.transform.parent = null;
        fenderBackRight.transform.parent = null;
    }
    private static void InstantiateWheels(GameObject wheelPrefab, int wheelNum, GameObject partition)
    {
        // Valid (x,z)-positions to sample from
        var xzPositions = new List<(float, float)>
        {
            (-0.07f, 0.05f), (-0.07f, -0.03f),
            (0f, 0.01f), (0f, -0.07f),
            (0.07f, 0.05f), (0.07f, -0.03f)
        };
        var randomIndices = Enumerable.Range(0, 6).ToList();
        Shuffle(randomIndices);

        var generatedWheels = new List<GameObject>();
        for (var i = 0; i < wheelNum; i++)
        {
            // Instantiate provided wheel type with generalized name
            var wheel = Instantiate(wheelPrefab, partition.transform);
            wheel.name = $"wheel_{i}";

            // Add RigidBody component for physical interaction with environment
            wheel.AddComponent<Rigidbody>();
        
            // Define position & rotation w.r.t. partition coordinate
            var xzPos = xzPositions[randomIndices[i]]; 
            wheel.transform.localPosition = new Vector3(
                xzPos.Item1, 0.03f, xzPos.Item2
            );
            wheel.transform.eulerAngles = new Vector3(
                0f, 0f, (Random.Range(0, 2) * 2 - 1) * 90f
            );
            
            generatedWheels.Add(wheel);
        }

        // Apply random rotations to the parent partition object
        var rotY = 90f * Random.Range(0, 4) + Random.Range(-10f, 10f);
        partition.transform.eulerAngles = new Vector3(0f, rotY, 0f);
        
        // Detach instantiated parts from parent
        foreach (var wheel in generatedWheels) wheel.transform.parent = null;
    }
    private static void InstantiateBolts(GameObject boltPrefab, int boltNum, GameObject partition)
    {
        // Valid (x,z)-positions to sample from
        var xzPositions = new List<(float, float)>
        {
            (-0.06f, -0.03f), (-0.03f, -0.03f), (0f, -0.03f), (0.03f, -0.03f), (0.06f, -0.03f),
            (-0.075f, 0.03f), (-0.045f, 0.03f), (-0.015f, 0.03f),
            (0.015f, 0.03f), (0.045f, 0.03f), (0.075f, 0.03f)
        };
        var randomIndices = Enumerable.Range(0, 11).ToList();
        Shuffle(randomIndices);

        var generatedBolts = new List<GameObject>();
        for (var i = 0; i < boltNum; i++)
        {
            // Instantiate provided wheel type with generalized name
            var bolt = Instantiate(boltPrefab, partition.transform);
            bolt.name = $"bolt_{i}";

            // Add RigidBody component for physical interaction with environment
            bolt.AddComponent<Rigidbody>();
        
            // Define position & rotation w.r.t. partition coordinate
            var xzPos = xzPositions[randomIndices[i]]; 
            bolt.transform.localPosition = new Vector3(
                xzPos.Item1, 0.03f, xzPos.Item2
            );
            bolt.transform.eulerAngles = new Vector3(
                (Random.Range(0, 2) * 2 - 1) * 90f, 0f, Random.Range(-10f, 10f)
            );
            
            generatedBolts.Add(bolt);
        }

        // Apply random rotations to the parent partition object
        var rotY = Random.Range(0f, 359.9f);
        partition.transform.eulerAngles = new Vector3(0f, rotY, 0f);
        
        // Detach instantiated parts from parent
        foreach (var bolt in generatedBolts) bolt.transform.parent = null;
    }

    // Helper method for randomly shuffling a list (Fisher-Yates shuffle)
    private static void Shuffle<T>(List<T> toShuffle)
    {
        for (var i = 0; i < toShuffle.Count-1; i++)
        {
            var temp = toShuffle[i];
            var randomIndex = Random.Range(i, toShuffle.Count);
            toShuffle[i] = toShuffle[randomIndex];
            toShuffle[randomIndex] = temp;
        }
    }
}
