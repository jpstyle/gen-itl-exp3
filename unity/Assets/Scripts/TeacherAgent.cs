using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.SideChannels;
using UnityEngine.Assertions;
using UnityEngine.Perception.GroundTruth.LabelManagement;
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
        new Vector3(-0.36f, 0.76f, 0.24f),
        new Vector3(0.36f, 0.76f, 0.24f),
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
                .OrderBy(x => x.Key.Item2)
                .Select(x => x.Value).ToList(),
            partitions[5],
            "t"
        );
        InstantiateBolts(targetTypes
                .Where(x => x.Key.Item1 == "bolt")
                .OrderBy(x => x.Key.Item2)
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
            if (actionBuffers.DiscreteActions[1] == 0)
                // Just 'Utter' action only
                StartCoroutine(Utter());
            else
                // Both 'Utter' action and some physical action queued; utter first
                // then act (in order to send any mask annotations based on 'before'
                // state)
                StartCoroutine(UtterThenAct(actionBuffers.DiscreteActions[1]));
        }
        else
        {
            // No 'Utter' action needed, just Act
            if (actionBuffers.DiscreteActions[1] != 0)
                StartCoroutine(Act(actionBuffers.DiscreteActions[1]));
        }
    }

    // ReSharper disable Unity.PerformanceAnalysis
    private IEnumerator Act(int actionType)
    {
        // Coroutine that executes specified physical action
        switch (actionType)
        {
            case 1:
            case 2:
                // PickUpLeft/Right action, parameter: (target object)
                var targetName = actionParameterBuffer.Dequeue();
                var targetEnt = EnvEntity.FindByObjectPath($"/{targetName}");
                var withLeft = actionType % 2 == 1;
                PickUp(targetEnt.gameObject, withLeft);
                break;
            case 3:
            case 4:
                // DropLeft/Right action, parameter: ()
                var fromLeft = actionType % 2 == 1;
                Drop(fromLeft);
                break;
            case 5:
            case 6:
                // AssembleRtoL/LtoR action, parameter: {contact point L, contact point R,
                // resultant subassembly string name)
                var leftPoint = actionParameterBuffer.Dequeue();
                var rightPoint = actionParameterBuffer.Dequeue();
                var productName = actionParameterBuffer.Dequeue();
                var rightToLeft = actionType % 2 == 1;
                Assemble(leftPoint, rightPoint, productName, rightToLeft);
                break;
            case 7:
                // InspectLeft action, parameter: ()
                break;
            case 8:
                // InspectRight action, parameter: ()
                break;
        }

        // All parameters consumed and none left in buffer
        Assert.IsTrue(actionParameterBuffer.Count == 0);
        yield break;
    }

    private IEnumerator UtterThenAct(int actionType)
    {
        // Coroutine that first invokes Utter(), waits until it finishes, and then
        // execute specified physical action
        // ReSharper disable once Unity.PerformanceCriticalCodeInvocation
        yield return StartCoroutine(Utter());
        yield return StartCoroutine(Act(actionType));
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

    private void PickUp(GameObject targetObj, bool withLeft)
    {
        // Pick up a target object on the tabletop with the specified hand
        var activeHand = withLeft ? leftHand : rightHand;

        // Move target object to hand position, reassign hand as the parent,
        // disable physics interaction
        targetObj.transform.parent = activeHand.transform;
        targetObj.transform.localPosition = Vector3.zero;
        var objRigidbody = targetObj.GetComponent<Rigidbody>();
        objRigidbody.isKinematic = true;
        objRigidbody.detectCollisions = false;
    }

    private void Drop(bool fromLeft)
    {
        // Drop an object currently held in the specified hand onto the tabletop
        var activeHand = fromLeft ? leftHand : rightHand;

        // Get handle of object (implicit assumption; max one item can be held per
        // manipulator)
        GameObject heldObj = null;
        foreach (Transform tr in activeHand.transform)
        {
            heldObj = tr.gameObject;
            break;
        }
        Assert.IsNotNull(heldObj);

        // Move target object above (y) some random x/z-position within the main
        // work area partition, 'release' by nullifying the hand's child status,
        float xPos, zPos;
        if (heldObj.name == "truck")
        {
            // Assuming the tabletop is pretty much clear now...
            xPos = 0f;
            zPos = _mainPartitionPosition.z + 0.12f;
        }
        else
        {
            xPos = _mainPartitionPosition.x + Random.Range(-0.21f, 0.21f);
            zPos = _mainPartitionPosition.z + Random.Range(-0.04f, 0.12f);
        }
        activeHand.localEulerAngles = Vector3.zero;
        var volume = GetBoundingVolume(heldObj);
        activeHand.localEulerAngles = fromLeft ? leftOriginalEuler : rightOriginalEuler;
        var yPos = _mainPartitionPosition.y + volume.extents.y + 0.06f;
        heldObj.transform.parent = null;
        heldObj.transform.position = new Vector3(xPos, yPos, zPos);

        // Re-enable physics & fast-forward physics simulation until the object rests
        // on the desktop
        var objRigidbody = heldObj.GetComponent<Rigidbody>();
        objRigidbody.isKinematic = false;
        objRigidbody.detectCollisions = true;
        Physics.simulationMode = SimulationMode.Script;
        for (var i=0; i<2000; i++)
        {
            Physics.Simulate(Time.fixedDeltaTime);
        }
        Physics.simulationMode = SimulationMode.FixedUpdate;
    }

    private void Assemble(
        string leftPoint, string rightPoint, string productName, bool rightToLeft
    )
    {
        // Assemble two subassemblies held in each hand as guided by the specified
        // target contact points, left-to-right or right-to-left
        var leftTargetParsed = leftPoint.Split("/");
        var rightTargetParsed = rightPoint.Split("/");

        // Source & target hands and points appropriately determined by `rightToLeft` parameter
        Transform srcHand, tgtHand;
        string srcAtomicPart, tgtAtomicPart, srcPointName, tgtPointName;
        if (rightToLeft)
        {
            srcHand = rightHand; tgtHand = leftHand;
            srcAtomicPart = rightTargetParsed[0];
            tgtAtomicPart = leftTargetParsed[0];
            srcPointName = $"cp_{rightTargetParsed[1]}";
            tgtPointName = $"cp_{leftTargetParsed[1]}";
        }
        else
        {
            srcHand = leftHand; tgtHand = rightHand;
            srcAtomicPart = leftTargetParsed[0];
            tgtAtomicPart = rightTargetParsed[0];
            srcPointName = $"cp_{leftTargetParsed[1]}";
            tgtPointName = $"cp_{rightTargetParsed[1]}";
        }

        Transform srcPoint = null; Transform tgtPoint = null;
        foreach (var candidateEnt in srcHand.GetComponentsInChildren<EnvEntity>())
        {
            if (candidateEnt.gameObject.name != srcAtomicPart) continue;
            foreach (Transform candidatePt in candidateEnt.gameObject.transform)
            {
                if (candidatePt.gameObject.name != srcPointName) continue;
                srcPoint = candidatePt;
                break;
            }
            if (srcPoint is not null) break;
        }
        foreach (var candidateEnt in tgtHand.GetComponentsInChildren<EnvEntity>())
        {
            if (candidateEnt.gameObject.name != tgtAtomicPart) continue;
            foreach (Transform candidatePt in candidateEnt.gameObject.transform)
            {
                if (candidatePt.gameObject.name != tgtPointName) continue;
                tgtPoint = candidatePt;
                break;
            }
            if (tgtPoint is not null) break;
        }
        Assert.IsNotNull(srcPoint); Assert.IsNotNull(tgtPoint);

        // Get relative pose (position & rotation) from source to target points, then
        // move source hand to target pose (rotation first, translation next)
        var relativeRotation = tgtPoint.rotation * Quaternion.Inverse(srcPoint.rotation);
        srcHand.rotation = relativeRotation * srcHand.rotation;
        var relativePosition = tgtPoint.position - srcPoint.position;
        srcHand.position += relativePosition;

        // Handles to subassembly objects held in source & target hands
        GameObject srcHeld = null; GameObject tgtHeld = null;
        foreach (Transform tr in srcHand.transform)
        {
            srcHeld = tr.gameObject;
            break;
        }
        foreach (Transform tr in tgtHand.transform)
        {
            tgtHeld = tr.gameObject;
            break;
        }
        Assert.IsNotNull(srcHeld); Assert.IsNotNull(tgtHeld);

        // Obtain pre-assembly bounding volume for later repositioning of children parts
        tgtHand.localEulerAngles = Vector3.zero;
        var beforeVolume = GetBoundingVolume(tgtHeld);
        tgtHand.localEulerAngles = rightToLeft ? leftOriginalEuler : rightOriginalEuler;

        // Merge the two subassemblies, 'releasing' from source hand by reassigning
        // parent transforms of parts in the source subassembly
        var childrenParts = new List<Transform>();
        foreach (Transform tr in srcHeld.transform) childrenParts.Add(tr);
        foreach (var tr in childrenParts) tr.parent = tgtHeld.transform;

        // Obtain post-assembly bounding volume for later repositioning of children parts
        tgtHand.localEulerAngles = Vector3.zero;
        var afterVolume = GetBoundingVolume(tgtHeld);
        tgtHand.localEulerAngles = rightToLeft ? leftOriginalEuler : rightOriginalEuler;

        // Finish merging by destroying source subassembly and renaming target subassembly
        // with the provided string name
        srcHeld.GetComponent<EnvEntity>().enabled = false;
        Destroy(srcHeld);
        tgtHeld.name = productName;

        // Repositioning children prefabs in merged subassembly so the center of the
        // bounding volume becomes origin of the local space
        foreach (Transform child in tgtHeld.transform)
            child.position += beforeVolume.center - afterVolume.center;

        // Move back the source hand to the original pose
        srcHand.localPosition = rightToLeft ? rightOriginalPosition : leftOriginalPosition;
        srcHand.localEulerAngles = rightToLeft ? rightOriginalEuler : leftOriginalEuler;
    }

    private static Bounds GetBoundingVolume(GameObject subassembly)
    {
        // Obtain encasing bounding volume of a subassembly GameObject (in world coordinates)
        // Obtain bounding volume that encases all descendant meshes
        var minCoords = Vector3.positiveInfinity;
        var maxCoords = Vector3.negativeInfinity;
        foreach (var mesh in subassembly.GetComponentsInChildren<MeshRenderer>())
        {
            // Update min/max coordinates of mesh bounds to obtain encasing volume later
            var meshBounds = mesh.bounds;
            minCoords = Vector3.Min(minCoords, meshBounds.min);
            maxCoords = Vector3.Max(maxCoords, meshBounds.max);
        }
        var boundingVolume = new Bounds((minCoords + maxCoords) / 2, maxCoords - minCoords);

        return boundingVolume;
    }

    private static void InstantiateAtomicPrefab(
        GameObject prefab, Material colorMaterial, GameObject partition,
        string wrapperName, string instanceName,
        Vector3 wrapperPos, Vector3 wrapperRot, Vector3 prtRot
    )
    {
        // Helper methods for initializing sampled parts on sampled locations

        // Create an empty wrapper GameObject representing 'singleton subassembly'
        var wrapper = new GameObject(wrapperName, typeof(Labeling))
        {
            transform =
            {
                position = partition.transform.position,
                parent = partition.transform
            }
        };
        var wrapperLabeling = wrapper.GetComponent<Labeling>();
        wrapperLabeling.labels.Add("subassembly");

        // Instantiate provided cabin type with generalized name
        var prefabInstance = Instantiate(prefab, wrapper.transform);
        prefabInstance.name = instanceName;

        // Apply color to colorable meshes 
        foreach (var mesh in prefabInstance.GetComponentsInChildren<MeshRenderer>())
        {
            if (colorMaterial is null) continue;
            if (mesh.material.name.StartsWith("Default"))
                mesh.material = colorMaterial;
        }

        // Make sure to attach EnvEntity component after adding the atomic child
        wrapper.AddComponent<EnvEntity>();

        // Reposition
        var boundingVolume = GetBoundingVolume(wrapper);
        var disposition = prefabInstance.transform.position - boundingVolume.center;
        prefabInstance.transform.position += disposition;

        // Then tweak the y-position of the wrapper so that they will float slightly above
        // the tabletop
        wrapperPos.y += boundingVolume.extents.y + 0.015f;

        // Add RigidBody component for physical interaction with environment
        wrapper.AddComponent<Rigidbody>();

        wrapper.transform.localPosition = wrapperPos;
        wrapper.transform.eulerAngles = wrapperRot;
        partition.transform.eulerAngles = prtRot;

        // Detach the wrapper from partition parent
        wrapper.transform.parent = null;
        partition.transform.eulerAngles = Vector3.zero;
    }
    private void InstantiateCabin(
        int typeIndex, GameObject partition, int colorIndex, string identifier
    )
    {
        // Define position w.r.t. partition coordinate
        var xPosition = Random.Range(-0.04f, 0.04f);
        var zPosition = Random.Range(-0.04f, 0.04f);
        var wrapperPos = new Vector3(xPosition, 0f, zPosition);

        // Apply random rotations to the parent partition object
        var rotY = Random.Range(0f, 359.9f);
        var prtRot = new Vector3(0f, rotY, 0f);

        InstantiateAtomicPrefab(
            cabinTypes[typeIndex], colors[colorIndex], partition,
            $"{identifier}_cabin_0", "cabin",
            wrapperPos, Vector3.zero, prtRot
        );
    }
    private void InstantiateLoad(int typeIndex, GameObject partition, string identifier)
    {
        // Define position w.r.t. partition coordinate
        var wrapperPos = Vector3.zero;

        // Apply random rotations to the parent partition object
        var rotY = 180f * Random.Range(0, 2) + Random.Range(-10f, 10f);
        var prtRot = new Vector3(0f, rotY, 0f);

        InstantiateAtomicPrefab(
            loadTypes[typeIndex], null, partition,
            $"{identifier}_load_0", "load",
            wrapperPos, Vector3.zero, prtRot
        );
    }
    private void InstantiateChassisFB(
        List<int> typeIndices, GameObject partition, string identifier
    )
    {
        // Define positions w.r.t. partition coordinate
        var wrapperPos = new List<Vector3>
        {
            new (0f, 0f, -0.05f),
            new (0f, 0f, 0.05f)
        };

        // Apply random rotations to the parent partition object
        var rotY = 90f * Random.Range(0, 4) + Random.Range(-15f, 15f);
        var prtRot = new Vector3(0f, rotY, 0f);

        // Create empty wrapper GameObjects representing 'singleton subassembly'
        var instantiateConfigs = new List<(GameObject, int, string, string)>
        {
            (frontChassisTypes[typeIndices[0]], 0, $"{identifier}_chassis_front_0", "chassis_front"),
            (backChassisTypes[typeIndices[1]], 1, $"{identifier}_chassis_back_0", "chassis_back")
        };
        instantiateConfigs
            .ForEach(config => InstantiateAtomicPrefab(
                config.Item1, null, partition,
                config.Item3, config.Item4,
                wrapperPos[config.Item2], Vector3.zero, prtRot
            ));
    }
    private void InstantiateChassisC(
        int typeIndex, GameObject partition, int centerColorIndex, string identifier
    )
    {
        // Define position w.r.t. partition coordinate
        var xPosition = Random.Range(-0.04f, 0.04f);
        var zPosition = Random.Range(-0.04f, 0.04f);
        var wrapperPos = new Vector3(xPosition, 0f, zPosition);

        // Apply random rotation to the parent partition object
        var rotY = Random.Range(-0f, 359.9f);
        var prtRot = new Vector3(0f, rotY, 0f);
        
        InstantiateAtomicPrefab(
            centerChassisTypes[typeIndex], colors[centerColorIndex], partition,
            $"{identifier}_chassis_center_0", "chassis_center",
            wrapperPos, Vector3.zero, prtRot
        );
    }
    private void InstantiateFenders(
        List<int> typeIndices, GameObject partition, List<int> colorIndices, string identifier
    )
    {
        // Define positions & rotations w.r.t. partition coordinate
        var xRotations = new List<float>
        {
            // -1 corresponds to x-rotation of -90, 1 corresponds to x-rotation of 90
            (Random.Range(0, 2) * 2 - 1) * 90f,
            (Random.Range(0, 2) * 2 - 1) * 90f,
            (Random.Range(0, 2) * 2 - 1) * 90f,
            (Random.Range(0, 2) * 2 - 1) * 90f
        };
        var zPositions = new List<float> {-0.09f, -0.03f, 0.03f, 0.09f};
        var randomIndices = Enumerable.Range(0, 4).ToList();
        Shuffle(randomIndices);
        var wrapperPos = new List<Vector3>
        {
            new (Random.Range(-0.03f, 0.03f), 0f, zPositions[randomIndices[0]]),
            new (Random.Range(-0.03f, 0.03f), 0f, zPositions[randomIndices[1]]),
            new (Random.Range(-0.03f, 0.03f), 0f, zPositions[randomIndices[2]]),
            new (Random.Range(-0.03f, 0.03f), 0f, zPositions[randomIndices[3]])
        };
        var wrapperRot = new List<Vector3>
        {
            new (xRotations[0], 0f, 0f),
            new (xRotations[1], 0f, 0f),
            new (xRotations[2], 0f, 0f),
            new (xRotations[3], 0f, 0f),
        };

        // Apply random rotations to the parent partition object
        var rotY = 90f * Random.Range(0, 4) + Random.Range(-5f, 5f);
        var prtRot = new Vector3(0f, rotY, 0f);

        var instantiateConfigs = new List<(GameObject, int, string, string)>
        {
            (fenderFrontLeftTypes[typeIndices[0]], 0, $"{identifier}_fl_fender_0", "fl_fender"),
            (fenderFrontRightTypes[typeIndices[1]], 1, $"{identifier}_fr_fender_0", "fr_fender"),
            (fenderBackLeftTypes[typeIndices[2]], 2, $"{identifier}_bl_fender_0", "bl_fender"),
            (fenderBackRightTypes[typeIndices[3]], 3, $"{identifier}_br_fender_0", "br_fender")
        };
        instantiateConfigs
            .ForEach(config => 
                InstantiateAtomicPrefab(
                    config.Item1, colors[colorIndices[config.Item2]], partition,
                    config.Item3, config.Item4,
                    wrapperPos[config.Item2], wrapperRot[config.Item2], prtRot
                ));
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

        // Apply random rotations to the parent partition object
        var rotY = 90f * Random.Range(0, 4) + Random.Range(-10f, 10f);
        var prtRot = new Vector3(0f, rotY, 0f);

        for (var i = 0; i < typeIndices.Count; i++)
        {
            // Define position & rotation w.r.t. partition coordinate
            var xzPos = xzPositions[randomIndices[i]]; 
            var wrapperPos = new Vector3(xzPos.Item1, 0f, xzPos.Item2);
            var wrapperRot = new Vector3(0f, 0f, (Random.Range(0, 2) * 2 - 1) * 90f);

            InstantiateAtomicPrefab(
                wheelTypes[typeIndices[i]], null, partition,
                $"{identifier}_wheel_{i}", "wheel",
                wrapperPos, wrapperRot, prtRot
            );
        }
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

        // Apply random rotations to the parent partition object
        var rotY = Random.Range(0f, 359.9f);
        var prtRot = new Vector3(0f, rotY, 0f);

        for (var i = 0; i < typeIndices.Count; i++)
        {
            // Define position & rotation w.r.t. partition coordinate
            var xzPos = xzPositions[randomIndices[i]]; 
            var wrapperPos = new Vector3(xzPos.Item1, 0f, xzPos.Item2);
            var wrapperRot = new Vector3(
                (Random.Range(0, 2) * 2 - 1) * 90f, 0f, Random.Range(-10f, 10f)
            );

            InstantiateAtomicPrefab(
                boltTypes[typeIndices[i]], null, partition,
                $"{identifier}_bolt_{i}", "bolt",
                wrapperPos, wrapperRot, prtRot
            );
        }
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
