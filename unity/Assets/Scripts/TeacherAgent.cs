using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.SideChannels;
using Random = UnityEngine.Random;

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

        var typeIndices = new Dictionary<(string, int), int>();
        var colorIndices = new Dictionary<(string, int), int>();
        foreach (var key in envParams.Keys())
        {
            var keyFields = key.Split("/");
            var partSupertype = keyFields[0];
            var descriptor = keyFields[1];
            var identifier = Convert.ToInt32(keyFields[2]);

            if (descriptor == "type")
                typeIndices[(partSupertype, identifier)] = GetEnvParam(envParams, key);
            else
                colorIndices[(partSupertype, identifier)] = GetEnvParam(envParams, key);
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
        InstantiateCabin(typeIndices[("cabin", 0)], partitions[0], colorIndices[("cabin", 0)]);
        InstantiateLoad(typeIndices[("load", 0)], partitions[1]);
        InstantiateChassis(
            new List<int>
            {
                typeIndices[("chassis_front", 0)],
                typeIndices[("chassis_center", 0)],
                typeIndices[("chassis_back", 0)]
            },
            partitions[2],
            colorIndices[("chassis_center", 0)]
        );
        InstantiateFenders(
            new List<int>
            {
                typeIndices[("fl_fender", 0)],
                typeIndices[("fr_fender", 0)],
                typeIndices[("bl_fender", 0)],
                typeIndices[("br_fender", 0)]
            },
            partitions[3],
            new List<int>
            {
                colorIndices[("fl_fender", 0)],
                colorIndices[("fr_fender", 0)],
                colorIndices[("bl_fender", 0)],
                colorIndices[("br_fender", 0)]
            }
        );
        InstantiateWheels(
            typeIndices
                .Where(x => x.Key.Item1 == "wheel")
                .Select(x => x.Value).ToList(),
            partitions[4]
        );
        InstantiateBolts(typeIndices
                .Where(x => x.Key.Item1 == "bolt")
                .Select(x => x.Value).ToList(),
            partitions[5]
        );

        // Cleanup partitions now they're no longer needed
        foreach (var partition in partitions) Destroy(partition);

        // Fast-forward physics simulation until generated objects rest on the desktop
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

    protected override List<(string, List<string>)> SubtypeOrderings()
    {
        return new List<(string, List<string>)>
        {
            ("cabin", cabinTypes.Select(t => t.name).ToList()),
            ("load", loadTypes.Select(t => t.name).ToList()),
            ("chassis_front", frontChassisTypes.Select(t => t.name).ToList()),
            ("chassis_center", centerChassisTypes.Select(t => t.name).ToList()),
            ("chassis_back", backChassisTypes.Select(t => t.name).ToList()),
            ("fl_fender", fenderFrontLeftTypes.Select(t => t.name).ToList()),
            ("fr_fender", fenderFrontRightTypes.Select(t => t.name).ToList()),
            ("bl_fender", fenderBackLeftTypes.Select(t => t.name).ToList()),
            ("br_fender", fenderBackRightTypes.Select(t => t.name).ToList()),
            ("wheel", wheelTypes.Select(t => t.name).ToList()),
            ("bolt", boltTypes.Select(t => t.name).ToList()),
            ("color", colors.Select(t => t.name.Replace("plastic_", "")).ToList())
        };
    }

    // Helper methods for initializing sampled parts on sampled locations
    private void InstantiateCabin(
        int typeIndex, GameObject partition, int colorIndex
    )
    {
        // Instantiate provided cabin type with generalized name
        var cabinPrefab = cabinTypes[typeIndex];
        var cabin = Instantiate(cabinPrefab, partition.transform);
        cabin.name = "cabin";

        // Apply sampled color
        foreach (var mesh in cabin.GetComponentsInChildren<MeshRenderer>())
            if (mesh.material.name.StartsWith("Default"))
                mesh.material = colors[colorIndex];

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
    private void InstantiateLoad(int typeIndex, GameObject partition)
    {
        // Instantiate provided load type with generalized name
        var loadPrefab = loadTypes[typeIndex];
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
    private void InstantiateChassis(
        List<int> typeIndices, GameObject partition, int centerColorIndex
    )
    {
        // Instantiate provided chassis types with generalized names
        var frontPrefab = frontChassisTypes[typeIndices[0]];
        var centerPrefab = centerChassisTypes[typeIndices[1]];
        var backPrefab = backChassisTypes[typeIndices[2]];
        var chassisFront = Instantiate(frontPrefab, partition.transform);
        var chassisCenter = Instantiate(centerPrefab, partition.transform);
        var chassisBack = Instantiate(backPrefab, partition.transform);
        chassisFront.name = "chassis_front";
        chassisCenter.name = "chassis_center";
        chassisBack.name = "chassis_back";

        // Apply sampled color to center chassis
        foreach (var mesh in chassisCenter.GetComponentsInChildren<MeshRenderer>())
            if (mesh.material.name.StartsWith("Default"))
                mesh.material = colors[centerColorIndex];

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
    private void InstantiateFenders(
        List<int> typeIndices, GameObject partition, List<int> colorIndices
    )
    {
        // Instantiate provided fender types with generalized names
        var frontLeftPrefab = fenderFrontLeftTypes[typeIndices[0]];
        var frontRightPrefab = fenderFrontRightTypes[typeIndices[1]];
        var backLeftPrefab = fenderBackLeftTypes[typeIndices[2]];
        var backRightPrefab = fenderBackRightTypes[typeIndices[3]];
        var fenderFrontLeft = Instantiate(frontLeftPrefab, partition.transform);
        var fenderFrontRight = Instantiate(frontRightPrefab, partition.transform);
        var fenderBackLeft = Instantiate(backLeftPrefab, partition.transform);
        var fenderBackRight = Instantiate(backRightPrefab, partition.transform);
        fenderFrontLeft.name = "fl_fender";
        fenderFrontRight.name = "fr_fender";
        fenderBackLeft.name = "bl_fender";
        fenderBackRight.name = "br_fender";

        // Apply sampled color to each fender
        foreach (var mesh in fenderFrontLeft.GetComponentsInChildren<MeshRenderer>())
            if (mesh.material.name.StartsWith("Default"))
                mesh.material = colors[colorIndices[0]];
        foreach (var mesh in fenderFrontRight.GetComponentsInChildren<MeshRenderer>())
            if (mesh.material.name.StartsWith("Default"))
                mesh.material = colors[colorIndices[1]];
        foreach (var mesh in fenderBackLeft.GetComponentsInChildren<MeshRenderer>())
            if (mesh.material.name.StartsWith("Default"))
                mesh.material = colors[colorIndices[2]];
        foreach (var mesh in fenderBackRight.GetComponentsInChildren<MeshRenderer>())
            if (mesh.material.name.StartsWith("Default"))
                mesh.material = colors[colorIndices[3]];

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
    private void InstantiateWheels(List<int> typeIndices, GameObject partition)
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
        for (var i = 0; i < typeIndices.Count; i++)
        {
            // Instantiate provided wheel type with generalized name
            var wheelPrefab = wheelTypes[typeIndices[i]];
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
    private void InstantiateBolts(List<int> typeIndices, GameObject partition)
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
        for (var i = 0; i < typeIndices.Count; i++)
        {
            // Instantiate provided wheel type with generalized name
            var boltPrefab = boltTypes[typeIndices[i]];
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
    
    // Environment parameter fetching logic is a bit lengthy to repeat, make shortcut
    private static int GetEnvParam(EnvironmentParameters envParams, string key)
    {
        return (int) envParams.GetWithDefault(key, 0f);
    }
}
