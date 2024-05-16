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
        new Vector3(-0.48f, 0.76f, 0.48f),
        new Vector3(-0.24f, 0.76f, 0.48f),
        new Vector3(0f, 0.76f, 0.48f),
        new Vector3(0.24f, 0.76f, 0.48f),
        new Vector3(0.48f, 0.76f, 0.48f),
        new Vector3(-0.42f, 0.88f, 0.75f),
        new Vector3(-0.14f, 0.88f, 0.75f),
        new Vector3(0.14f, 0.88f, 0.75f),
        new Vector3(0.42f, 0.88f, 0.75f)
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

        var partGroups = new Dictionary<string, string>
        {
            { "cabin", "cabin" }, { "load", "load" },
            { "chassis_front", "chassis_fb" },
            { "chassis_center", "chassis_center" },
            { "chassis_back", "chassis_fb" },
            { "fl_fender", "fender" }, { "fr_fender", "fender" },
            { "bl_fender", "fender" }, { "br_fender", "fender" },
            { "wheel", "wheel" }, { "bolt", "bolt" }
        };
        var targetTypes = new Dictionary<(string, string), int>();
        var targetColors = new Dictionary<(string, string), int>();
        var targetGroups = new List<string>();
        var distractorTypes = new Dictionary<(string, string), int>();
        var distractorColors = new Dictionary<(string, string), int>();
        var distractorGroups = new List<string>();
        foreach (var key in envParams.Keys())
        {
            var keyFields = key.Split("/");
            var partSupertype = keyFields[0];
            var descriptor = keyFields[1];
            var identifier = keyFields[2];

            var groupType = partGroups[partSupertype];

            var paramValue = GetEnvParam(envParams, key);
            if (identifier.StartsWith("t"))
            {
                // Parts that make up desired target object
                if (descriptor == "type")
                    targetTypes[(partSupertype, identifier)] = paramValue;
                else
                    targetColors[(partSupertype, identifier)] = paramValue;
                
                if (!targetGroups.Contains(groupType))
                    targetGroups.Add(groupType);
            }
            else
            {
                // Parts that act as distractors
                if (descriptor == "type")
                    distractorTypes[(partSupertype, identifier)] = paramValue;
                else
                    distractorColors[(partSupertype, identifier)] = paramValue;

                if (!distractorGroups.Contains(groupType))
                    distractorGroups.Add(groupType);
            }
        }

        // Create appropriate number of 'workspace partitions'; square areas that take up
        // space on the tabletop (corresponding gameObjects don't need to be Planes though,
        // we only need reference coordinates)
        var numPartitions = targetGroups.Count + distractorGroups.Count;
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
        InstantiateCabin(
            targetTypes[("cabin", "t0")],
            partitions[0], 
            targetColors[("cabin", "t0")],
            "t"
        );
        InstantiateLoad(targetTypes[("load", "t0")], partitions[1], "t");
        InstantiateChassisFB(
            new List<int>
            {
                targetTypes[("chassis_front", "t0")],
                targetTypes[("chassis_back", "t0")]
            },
            partitions[2],
            "t"
        );
        InstantiateChassisC(
            targetTypes[("chassis_center", "t0")],
            partitions[3],
            targetColors[("chassis_center", "t0")],
            "t"
        );
        InstantiateFenders(
            new List<int>
            {
                targetTypes[("fl_fender", "t0")],
                targetTypes[("fr_fender", "t0")],
                targetTypes[("bl_fender", "t0")],
                targetTypes[("br_fender", "t0")]
            },
            partitions[4],
            new List<int>
            {
                targetColors[("fl_fender", "t0")],
                targetColors[("fr_fender", "t0")],
                targetColors[("bl_fender", "t0")],
                targetColors[("br_fender", "t0")]
            },
            "t"
        );
        InstantiateWheels(
            targetTypes
                .Where(x => x.Key.Item1 == "wheel")
                .Select(x => x.Value).ToList(),
            partitions[5],
            "t"
        );
        InstantiateBolts(targetTypes
                .Where(x => x.Key.Item1 == "bolt")
                .Select(x => x.Value).ToList(),
            partitions[6],
            "t"
        );

        for (var i=0; i<distractorGroups.Count; i++)
        {
            var groupType = distractorGroups[i];
            var partition = partitions[i+7];
            switch (groupType)
            {
                case "cabin":
                    InstantiateCabin(
                        distractorTypes[("cabin", "d0")],
                        partition,
                        distractorColors[("cabin", "d0")],
                        "d"
                    );
                    break;
                case "load":
                    InstantiateLoad(distractorTypes[("load", "d0")], partition, "d");
                    break;
                case "chassis_center":
                    InstantiateChassisC(
                        distractorTypes[("chassis_center", "d0")],
                        partition,
                        distractorColors[("chassis_center", "d0")],
                        "d"
                    );
                    break;
                case "fender":
                    InstantiateFenders(
                        new List<int>
                        {
                            distractorTypes[("fl_fender", "d0")],
                            distractorTypes[("fr_fender", "d0")],
                            distractorTypes[("bl_fender", "d0")],
                            distractorTypes[("br_fender", "d0")]
                        },
                        partition,
                        new List<int>
                        {
                            distractorColors[("fl_fender", "d0")],
                            distractorColors[("fr_fender", "d0")],
                            distractorColors[("bl_fender", "d0")],
                            distractorColors[("br_fender", "d0")]
                        },
                        "d"
                    );
                    break;
                case "wheel":
                    InstantiateWheels(
                        distractorTypes
                            .Where(x => x.Key.Item1 == "wheel")
                            .Select(x => x.Value).ToList(),
                        partition,
                        "d"
                    );
                    break;
                default:
                    // Shouldn't reach here
                    throw new Exception("Invalid distractor part type");
            }
        }

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
        int typeIndex, GameObject partition, int colorIndex, string identifier
    )
    {
        // Instantiate provided cabin type with generalized name
        var cabinPrefab = cabinTypes[typeIndex];
        var cabin = Instantiate(cabinPrefab, partition.transform);
        cabin.name = $"{identifier}_cabin";

        // Apply sampled color
        foreach (var mesh in cabin.GetComponentsInChildren<MeshRenderer>())
            if (mesh.material.name.StartsWith("Default"))
                mesh.material = colors[colorIndex];

        // Add RigidBody component for physical interaction with environment
        cabin.AddComponent<Rigidbody>();
        
        // Define position w.r.t. partition coordinate
        var xPosition = Random.Range(-0.04f, 0.04f);
        var zPosition = Random.Range(-0.04f, 0.04f);
        cabin.transform.localPosition = new Vector3(xPosition, 0.06f, zPosition);

        // Apply random rotations to the parent partition object
        var rotY = Random.Range(0f, 359.9f);
        partition.transform.eulerAngles = new Vector3(0f, rotY, 0f);
        
        // Detach instantiated part from parent
        cabin.transform.parent = null;
    }
    private void InstantiateLoad(int typeIndex, GameObject partition, string identifier)
    {
        // Instantiate provided load type with generalized name
        var loadPrefab = loadTypes[typeIndex];
        var load = Instantiate(loadPrefab, partition.transform);
        load.name = $"{identifier}_load";

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
    private void InstantiateChassisFB(
        List<int> typeIndices, GameObject partition, string identifier
    )
    {
        // Instantiate provided chassis types with generalized names
        var frontPrefab = frontChassisTypes[typeIndices[0]];
        var backPrefab = backChassisTypes[typeIndices[1]];
        var chassisFront = Instantiate(frontPrefab, partition.transform);
        var chassisBack = Instantiate(backPrefab, partition.transform);
        chassisFront.name = $"{identifier}_chassis_front";
        chassisBack.name = $"{identifier}_chassis_back";

        // Add RigidBody component for physical interaction with environment
        chassisFront.AddComponent<Rigidbody>();
        chassisBack.AddComponent<Rigidbody>();
        
        // Define positions w.r.t. partition coordinate
        chassisFront.transform.localPosition = new Vector3(0f, 0.02f, -0.05f);
        chassisBack.transform.localPosition = new Vector3(0f, 0.02f, 0.05f);

        // Apply random rotations to the parent partition object
        var rotY = 90f * Random.Range(0, 4) + Random.Range(-15f, 15f);
        partition.transform.eulerAngles = new Vector3(0f, rotY, 0f);
        
        // Detach instantiated parts from parent
        chassisFront.transform.parent = null;
        chassisBack.transform.parent = null;
    }
    private void InstantiateChassisC(
        int typeIndex, GameObject partition, int centerColorIndex, string identifier
    )
    {
        // Instantiate provided chassis type with generalized names
        var centerPrefab = centerChassisTypes[typeIndex];
        var chassisCenter = Instantiate(centerPrefab, partition.transform);
        chassisCenter.name = $"{identifier}_chassis_center";

        // Apply sampled color to center chassis
        foreach (var mesh in chassisCenter.GetComponentsInChildren<MeshRenderer>())
            if (mesh.material.name.StartsWith("Default"))
                mesh.material = colors[centerColorIndex];

        // Add RigidBody component for physical interaction with environment
        chassisCenter.AddComponent<Rigidbody>();
        
        // Define position w.r.t. partition coordinate
        var xPosition = Random.Range(-0.04f, 0.04f);
        var zPosition = Random.Range(-0.04f, 0.04f);
        chassisCenter.transform.localPosition = new Vector3(xPosition, 0.02f, zPosition);

        // Apply random rotation to the parent partition object
        var rotY = Random.Range(-0f, 359.9f);
        partition.transform.eulerAngles = new Vector3(0f, rotY, 0f);
        
        // Detach instantiated part from parent
        chassisCenter.transform.parent = null;
    }
    private void InstantiateFenders(
        List<int> typeIndices, GameObject partition, List<int> colorIndices, string identifier
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
        fenderFrontLeft.name = $"{identifier}_fl_fender";
        fenderFrontRight.name = $"{identifier}_fr_fender";
        fenderBackLeft.name = $"{identifier}_bl_fender";
        fenderBackRight.name = $"{identifier}_br_fender";

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
        var xRotations = new List<int>
        {
            // -1 corresponds to x-rotation of -90, requiring 0.01 z-position offset
            // 1 corresponds to x-rotation of 90, requiring -0.01 z-position offset
            Random.Range(0, 2) * 2 - 1,
            Random.Range(0, 2) * 2 - 1,
            Random.Range(0, 2) * 2 - 1,
            Random.Range(0, 2) * 2 - 1
        };
        var zPositions = new List<float> {-0.09f, -0.03f, 0.03f, 0.09f};
        var randomIndices = Enumerable.Range(0, 4).ToList();
        Shuffle(randomIndices);
        fenderFrontLeft.transform.localPosition = new Vector3(
            Random.Range(-0.03f, 0.03f), 0.02f,
            zPositions[randomIndices[0]] + xRotations[0] * -0.01f
        );
        fenderFrontRight.transform.localPosition = new Vector3(
            Random.Range(-0.03f, 0.03f), 0.02f,
            zPositions[randomIndices[1]] + xRotations[1] * -0.01f
        );
        fenderBackLeft.transform.localPosition = new Vector3(
            Random.Range(-0.03f, 0.03f), 0.02f,
            zPositions[randomIndices[2]] + xRotations[2] * -0.01f
        );
        fenderBackRight.transform.localPosition = new Vector3(
            Random.Range(-0.03f, 0.03f), 0.02f,
            zPositions[randomIndices[3]] + xRotations[3] * -0.01f
        );
        fenderFrontLeft.transform.eulerAngles = new Vector3(xRotations[0] * 90f, 0f, 0f);
        fenderFrontRight.transform.eulerAngles = new Vector3(xRotations[1] * 90f, 0f, 0f);
        fenderBackLeft.transform.eulerAngles = new Vector3(xRotations[2] * 90f, 0f, 0f);
        fenderBackRight.transform.eulerAngles = new Vector3(xRotations[3] * 90f, 0f, 0f);

        // Apply random rotations to the parent partition object
        var rotY = 90f * Random.Range(0, 4) + Random.Range(-5f, 5f);
        partition.transform.eulerAngles = new Vector3(0f, rotY, 0f);
        
        // Detach instantiated parts from parent
        fenderFrontLeft.transform.parent = null;
        fenderFrontRight.transform.parent = null;
        fenderBackLeft.transform.parent = null;
        fenderBackRight.transform.parent = null;
    }
    private void InstantiateWheels(
        List<int> typeIndices, GameObject partition, string identifier
    )
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
            wheel.name = $"{identifier}_wheel_{i}";

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
    private void InstantiateBolts(
        List<int> typeIndices, GameObject partition, string identifier
    )
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
            bolt.name = $"{identifier}_bolt_{i}";

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
