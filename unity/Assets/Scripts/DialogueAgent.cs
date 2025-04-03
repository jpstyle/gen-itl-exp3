using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Perception.GroundTruth;
using UnityEngine.Perception.GroundTruth.LabelManagement;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Random = UnityEngine.Random;

public class DialogueAgent : Agent
{
    // String identifier name for the dialogue participant agent
    public string dialogueParticipantID;

    // Commit utterances through this dialogue channel
    public DialogueChannel dialogueChannel;

    // Message communication buffer queues
    public readonly Queue<RecordData> incomingMsgBuffer = new();
    public readonly Queue<(string, string, Dictionary<(int, int), (EntityRef, bool)>)> outgoingMsgBuffer = new();

    // Stores queue of ground-truth mask info requests received from the backend
    public readonly Queue<string> gtMaskRequests = new();

    // Stores currently active calibration image request received from the backend
    [HideInInspector]
    public int calibrationImageRequest = -1;

    // Stores whether any part subtype ordering info request is received from the backend
    [HideInInspector]
    public bool subtypeOrderingRequest;     // false by default

    // Stores any action parameters of string type
    public readonly Queue<(string, EntityRef)> actionParameterBuffer = new();

    // For visualizing handheld objects
    public Transform leftHand;
    public Transform rightHand;
    // Key manipulator positions
    [HideInInspector]
    public Vector3 leftOriginalPosition;
    [HideInInspector]
    public Vector3 rightOriginalPosition;
    [HideInInspector]
    public Quaternion inspectOriginalRotation;

    // For 3D structure inspection
    public Transform relativeViewCenter;
    public Transform relativeViewPoint;

    // Communication side channel to Python backend for requesting decisions
    protected string channelUuid;
    protected MessageSideChannel backendMsgChannel;

    // Camera sensor & annotating perception camera component
    private CameraSensorComponent _cameraSensor;
    private PerceptionCamera _perCam;

    // Behavior type as MLAgent
    private BehaviorType _behaviorType;
    
    // Boolean flag indicating an Utter action is invoked as coroutine and currently
    // running; for preventing multiple invocation of Utter coroutine 
    private bool _uttering;
    // Analogous flag for Act action
    private bool _acting;
    // Analogous flag for CaptureMask method
    private bool _maskCapturing;

    // Boolean flag indicating whether there is a pending request for decision to be
    // made; set to true when a request is blocked by _uttering or _acting flag
    private bool _decisionRequestPending;

    // For controlling minimal update interval, to allow visual inspection during runs
    private float _nextTimeToAct;
    private const float TimeInterval = 0.1f;

    // Store workplace partition info for main work area
    private readonly Vector3 _mainPartitionPosition = new(0f, 0.76f, 0.24f);

    public void Start()
    {
        _cameraSensor = GetComponent<CameraSensorComponent>();
        _perCam = _cameraSensor.Camera.GetComponent<PerceptionCamera>();
        if (_perCam is null)
            throw new Exception(
                "This agent's camera sensor doesn't have a PerceptionCamera component"
            );

        _behaviorType = GetComponent<BehaviorParameters>().BehaviorType;
        _nextTimeToAct = Time.time;

        leftOriginalPosition = leftHand.localPosition;
        rightOriginalPosition = rightHand.localPosition;

        dialogueChannel.dialogueParticipants[dialogueParticipantID] = this;
    }

    public override void OnEpisodeBegin()
    {
        // Say anything this agent has to say
        StartCoroutine(Utter());
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

    public void Update()
    {
        if (Time.time < _nextTimeToAct) return;
        
        _nextTimeToAct += TimeInterval;
        if (_behaviorType == BehaviorType.HeuristicOnly)
        {
            // Always empty incoming message buffer and call RequestDecision
            incomingMsgBuffer.Clear();
            RequestDecision();
        }
        else
        {
            // Trying to consult backend for requesting decision only when needed, in order
            // to minimize communication of visual observation data
            var hasToDo = false;
            hasToDo |= incomingMsgBuffer.Count > 0;
            hasToDo |= calibrationImageRequest != -1;
            hasToDo |= subtypeOrderingRequest;
            if (!hasToDo & !_decisionRequestPending) return;

            // If unprocessed incoming messages exist, process and consult backend
            while (incomingMsgBuffer.Count > 0)
            {
                // Fetch single message record from queue
                var incomingMessage = incomingMsgBuffer.Dequeue();
                if (!incomingMessage) continue;     // Empty message: content-less ping

                // (If any) Translate EnvEntity reference by UID to segmentation mask w.r.t.
                // this agent's camera sensor
                var demRefs = new Dictionary<(int, int), EntityRef>();
                foreach (var (range, (entMask, entName)) in incomingMessage.demonstrativeReferences)
                    if (entMask is not null)
                        demRefs[range] = new EntityRef(entMask);
                    else
                    {
                        Assert.IsNotNull(entName);
                        demRefs[range] = new EntityRef(entName);
                    }

                // Send message via side channel
                backendMsgChannel.SendMessageToBackend(
                    incomingMessage.speaker, incomingMessage.utterance, demRefs
                );
            }

            // Handle any valid calibration image request by changing (camera) pose and
            // requesting decision (automatically sending visual observation)
            if (calibrationImageRequest is >= 0 and < 24)
            {
                // Hard-coding pose changes per request int id here...
                var rem4 = calibrationImageRequest % 4;
                var quot12 = calibrationImageRequest / 12;
                var quot4Rem3 = calibrationImageRequest / 4 % 3;
                var tx = quot4Rem3 switch
                {
                    0 => -0.25f,
                    1 => 0f,
                    _ => 0.25f
                };
                var tz = quot12 == 0 ? -0.8f : -0.7f;
                var rx = quot12 == 0 ? 30f : 45f;
                var ry = (quot4Rem3, quot12, rem4) switch
                {
                    (0, 0, 0) => 10f,
                    (0, 0, 1) => 20f,
                    (0, 0, 2) => 30f,
                    (0, 0, 3) => 40f,
                    (0, 1, 0) => 5f,
                    (0, 1, 1) => 15f,
                    (0, 1, 2) => 25f,
                    (0, 1, 3) => 35f,
                    (1, 0, 0) => -15f,
                    (1, 0, 1) => -5f,
                    (1, 0, 2) => 5f,
                    (1, 0, 3) => 15f,
                    (1, 1, 0) => -15f,
                    (1, 1, 1) => -5f,
                    (1, 1, 2) => 5f,
                    (1, 1, 3) => 15f,
                    (2, 0, 0) => -40f,
                    (2, 0, 1) => -30f,
                    (2, 0, 2) => -20f,
                    (2, 0, 3) => -10f,
                    (2, 1, 0) => -35f,
                    (2, 1, 1) => -25f,
                    (2, 1, 2) => -15f,
                    _ => -5f
                };

                // Set agent (x,z)-translation
                transform.position = new Vector3(tx, 0.85f, tz);
                // Set agent camera (x,y)-rotation
                _cameraSensor.Camera.transform.eulerAngles = new Vector3(rx, ry, 0f);
            }
            else if (calibrationImageRequest == 24)
            {
                // Deactivate chessboard pattern object
                // ReSharper disable once Unity.PerformanceCriticalCodeInvocation
                var chessboard = GameObject.Find("Chessboard");
                chessboard.SetActive(false);

                // Ensure default pose at the end of calibration image request signal
                // Set agent (x,z)-translation
                transform.position = new Vector3(0f, 0.85f, -0.3f);
                // Set agent camera (x,y)-rotation
                _cameraSensor.Camera.transform.eulerAngles = new Vector3(60f, 0f, 0f);

                // Turn off request flag
                calibrationImageRequest = -1;
            }

            // If any part subtype ordering info requests are pending, handle them here by
            // sending info to backend
            if (subtypeOrderingRequest)
            {
                var responseString = "# Subtype orderings response: ";
                var orderingInfo = SubtypeOrderings();

                var responseSubstrings = new List<string>();
                foreach (var (supertype, subtypes) in orderingInfo)
                {
                    var substring = $"{supertype} - "; 
                    substring += string.Join(", ", subtypes.ToArray());
                    responseSubstrings.Add(substring);
                }
                responseString += string.Join(" // ", responseSubstrings.ToArray());

                backendMsgChannel.SendMessageToBackend("System", responseString);
                subtypeOrderingRequest = false;    // Reset flag
            }

            // Now wait for decision, if not currently undergoing some action execution
            if (_uttering | _acting)
                _decisionRequestPending = true;
            else
            {
                RequestDecision();
                _decisionRequestPending = false;
            }
        }
    }

    protected IEnumerator CaptureAnnotations()
    {
        // If a coroutine invocation is still running, do not start another; else,
        // set flag
        if (_maskCapturing) yield break;
        _maskCapturing = true;

        // First send a request for a capture to the PerceptionCamera component of the
        // Camera to which the CameraSensorComponent is attached
        yield return null;      // Wait for a frame to render
        _perCam.RequestCapture();
        for (var i = 0; i < 5; i++)
            yield return null;
        // Waiting several more frames to ensure annotations were captured (This is
        // somewhat ad hoc indeed... but it works)

        // Wait until annotations are ready in the storage endpoint for retrieval
        yield return new WaitUntil(() => EnvEntity.annotationStorage.annotationsUpToDate);

        // Finally, update segmentation masks of all EnvEntity instances based on the data
        // stored in the endpoint
        EnvEntity.UpdateAnnotationsAll();

        // Reset flag on exit
        _maskCapturing = false;
    }

    private IEnumerator Utter()
    {
        // If a coroutine invocation is still running, do not start another; else,
        // set flag
        if (_uttering) yield break;
        _uttering = true;

        // Dequeue all messages to utter
        var messagesToUtter = new List<(string, string, Dictionary<(int, int), (EntityRef, bool)>)>();
        while (outgoingMsgBuffer.Count > 0)
            messagesToUtter.Add(outgoingMsgBuffer.Dequeue());
        
        // If no outgoing message to utter, can terminate here
        if (messagesToUtter.Count == 0 && gtMaskRequests.Count == 0)
        {
            _uttering = false;
            yield break;
        }

        // Ensuring annotations are up to date
        EnvEntity.annotationStorage.annotationsUpToDate = false;
        yield return StartCoroutine(CaptureAnnotations());

        // If any ground-truth mask requests are pending, handle them here by
        // sending info to backend
        if (gtMaskRequests.Count > 0)
        {
            var responseString = "# GT mask response: ";
            var responseMasks = new Dictionary<(int, int), EntityRef>();
            var stringPointer = responseString.Length;

            var partStrings = new List<string>();
            while (gtMaskRequests.Count > 0)
            {
                var req = gtMaskRequests.Dequeue();

                if (req == "*")
                {
                    // Requesting all existing (top-level) EnvEntities, enqueue them all
                    Assert.IsTrue(gtMaskRequests.Count == 0);   // Should be the only request
                    var topEntities = 
                        FindObjectsByType<EnvEntity>(FindObjectsSortMode.None)
                        .Where(
                            ent => ent.gameObject.transform.parent is null
                        )
                        .Select(ent => ent.gameObject.name)
                        .ToList();
                    topEntities.Sort();
                    foreach (var entName in topEntities) gtMaskRequests.Enqueue(entName);
                }
                else
                {
                    // Find relevant EnvEntity and fetch mask
                    var foundEnt = EnvEntity.FindByObjectPath("/" + req + "/" + req);
                    var gtMaskRef = new EntityRef(GetSensorMask(foundEnt));

                    // Part instance name + subtype string identifier; the latter is used
                    // as `identifying codes` for language-less player types
                    var nameWithType = req + "/" + foundEnt.licensedLabel; 
                    partStrings.Add(nameWithType);

                    var range = (stringPointer, stringPointer + nameWithType.Length);
                    responseMasks[range] = gtMaskRef;

                    stringPointer += nameWithType.Length;
                    if (gtMaskRequests.Count > 0) stringPointer += 2;
                        // Account for ", " delimiter
                }
            }
            responseString += string.Join(", ", partStrings.ToArray());

            backendMsgChannel.SendMessageToBackend(
                "System", responseString, responseMasks
            );
        }

        // Now utter individual messages
        foreach (var (speaker, utterance, demRefs) in messagesToUtter)
        {
            if (demRefs is not null && demRefs.Count > 0)
            {
                // Need to resolve demonstrative reference masks to corresponding EnvEntity (uid)
                var demRefsResolved = new Dictionary<(int, int), (float[], string)>();
                var targetDisplay = _cameraSensor.Camera.targetDisplay;

                foreach (var (range, (demRef, asMask)) in demRefs)
                {
                    EnvEntity ent;
                    switch (demRef.refType)
                    {
                        case EntityRefType.Mask:
                            var screenMask = MaskCoordinateSwitch(demRef.maskRef, false);
                            // ReSharper disable once Unity.PerformanceCriticalCodeInvocation
                            ent = EnvEntity.FindByMask(screenMask, targetDisplay);
                            break;
                        case EntityRefType.String:
                            ent = EnvEntity.FindByObjectPath(demRef.stringRef);
                            break;
                        default:
                            // Shouldn't reach here but anyway...
                            throw new Exception("Invalid reference data type?");
                    }

                    // Obtain mask at the point of utterance; by the time listeners process
                    // incoming messages, EnvEntity's GameObject may be already destroyed
                    demRefsResolved[range] = (
                        asMask ? GetSensorMask(ent) : null,
                        asMask ? null : ent.gameObject.name
                    );
                }

                dialogueChannel.CommitUtterance(dialogueParticipantID, utterance, demRefsResolved);
            }
            else
                // No demonstrative references to process and resolve
                dialogueChannel.CommitUtterance(dialogueParticipantID, utterance);
        }

        // Reset flag on exit
        _uttering = false;
    }

    // ReSharper disable Unity.PerformanceAnalysis
    private IEnumerator Act(int actionType)
    {
        // If a coroutine invocation is still running, do not start another; else,
        // set flag
        if (_acting) yield break;
        _acting = true;

        var actionEffect = "";

        // Coroutine that executes specified physical action
        var leftOrRight = actionType % 2 == 1;
        switch (actionType)
        {
            case 3:
            case 4:
                // PickUpLeft/Right action, parameter: {target object}, where target
                // may be designated by GameObject path or segmentation mask
                var pickUpParam1 = actionParameterBuffer.Dequeue();
                var targetName = pickUpParam1.Item1.Replace("str|", "");
                var targetEnt = EnvEntity.FindByObjectPath($"/{targetName}");
                var pickUpParam2 = actionParameterBuffer.Dequeue();
                var labelKey = pickUpParam2.Item1.Replace("str|", "");
                actionEffect = PickUp(targetEnt, labelKey, leftOrRight);
                break;
            case 5:
            case 6:
                // DropLeft/Right action, parameter: {}
                actionEffect = Drop(leftOrRight);
                break;
            case 7:
            case 8:
                // AssembleRtoL/LtoR action, two possible parameter signatures:
                //  1) {resultant subassembly name, object/contact L, object/contact R}
                //  2) {resultant subassembly name, manipulator transform}
                var assembleParam1 = actionParameterBuffer.Dequeue();
                var productName = assembleParam1.Item1.Replace("str|", "");
                
                var assembleParam2 = actionParameterBuffer.Dequeue();
                if (assembleParam2.Item1.StartsWith("str|"))
                {
                    var leftPoint = assembleParam2.Item1.Replace("str|", "");
                    var assembleParam3 = actionParameterBuffer.Dequeue();
                    var rightPoint = assembleParam3.Item1.Replace("str|", "");
                    actionEffect = Assemble(productName, leftPoint, rightPoint, leftOrRight);
                }
                else
                {
                    Assert.IsTrue(assembleParam2.Item1.StartsWith("floats|"));
                    var rotation = assembleParam2.Item1
                        .Replace("floats|", "").Split("/").Select(float.Parse).ToList();
                    var rotationQuaternion = new Quaternion(
                        rotation[1], rotation[2], rotation[3], rotation[0]
                    );          // wxyz to xyzw
                    var assembleParam3 = actionParameterBuffer.Dequeue();
                    var position = assembleParam3.Item1
                        .Replace("floats|", "").Split("/").Select(float.Parse).ToList();
                    var positionVector3 = new Vector3(
                        position[0], position[1], position[2]
                    );          // xyz
                    actionEffect = Assemble(
                        productName, rotationQuaternion, positionVector3, leftOrRight
                    );
                }
                break;
            case 9:
            case 10:
                // InspectLeft/Right action, parameter: {view angle index}
                var inspectParam1 = actionParameterBuffer.Dequeue();
                var inspectParam2 = actionParameterBuffer.Dequeue();
                var inspectedObjName = inspectParam1.Item1.Replace("str|", "");
                var viewAngleIndex = Convert.ToInt32(inspectParam2.Item1.Replace("int|", ""));
                actionEffect = Inspect(viewAngleIndex, inspectedObjName, leftOrRight);
                break;
            case 11:
            case 12:
                // DisassembleLeft/Right action, parameters: {resultant subassembly names on
                // left and right, parts to take away from the subassembly held}
                var disassembleParam1 = actionParameterBuffer.Dequeue();
                var disassembleParam2 = actionParameterBuffer.Dequeue();
                var disassembleParam3 = actionParameterBuffer.Dequeue();
                var leftName = disassembleParam1.Item1.Replace("str|", "");
                var rightName = disassembleParam2.Item1.Replace("str|", "");
                var numTakeawayParts = Convert.ToInt32(disassembleParam3.Item1.Replace("int|", ""));
                var takeawayParts = new List<string>();
                for (var i = 0; i < numTakeawayParts; i++)
                {
                    var disassembleParamNext = actionParameterBuffer.Dequeue();
                    takeawayParts.Add(disassembleParamNext.Item1.Replace("str|", ""));
                }
                actionEffect = Disassemble(leftName, rightName, takeawayParts, leftOrRight);
                break;
        }

        // All parameters consumed
        Assert.IsTrue(actionParameterBuffer.Count == 0);

        // Changes made to environment, Perception cameras need capture again
        EnvEntity.annotationStorage.annotationsUpToDate = false;
        yield return StartCoroutine(CaptureAnnotations());

        // Report action effect to self iff the learner agent has carried out this action
        if (actionEffect.Length > 0)
        {
            // To self and broadcast to others
            if (this is StudentAgent)
            {
                // Student: Recording action effect and send to self's backend
                backendMsgChannel.SendMessageToBackend(
                    dialogueParticipantID, actionEffect
                );
                // Send a null message as content-less ping, so that student will
                // request next step's decision from backend
                incomingMsgBuffer.Enqueue(null);
            }
            // Student & Teacher: Broadcast action effect to student
            dialogueChannel.CommitUtterance(
                dialogueParticipantID, actionEffect
            );
        }

        // Reset flag on exit
        _acting = false;
    }

    private IEnumerator UtterThenAct(int actionType)
    {
        // Coroutine that first invokes Utter(), waits until it finishes, and then
        // execute specified physical action
        // ReSharper disable once Unity.PerformanceCriticalCodeInvocation
        yield return StartCoroutine(Utter());
        yield return StartCoroutine(Act(actionType));
    }

    private string PickUp(EnvEntity targetEnt, string labelKey, bool withLeft)
    {
        var directionString = withLeft ? "left" : "right";

        // Action aftermath info to return; name of object picked up, pose of
        // manipulator, poses of atomic parts contained in the object picked up
        var actionEffect = $"# Effect: pick_up_{directionString}(";

        // Pick up a target object on the tabletop with the specified hand
        var activeHand = withLeft ? leftHand : rightHand;

        // Move target object to hand position, reassign hand as the parent,
        // disable physics interaction
        var targetObj = targetEnt.gameObject;
        targetObj.transform.parent = activeHand.transform;
        targetObj.transform.localPosition = Vector3.zero;
        var objRigidbody = targetObj.GetComponent<Rigidbody>();
        objRigidbody.isKinematic = true;
        objRigidbody.detectCollisions = false;

        // Report Unity-side gameObject name of the picked up entity
        actionEffect += targetObj.name + ",";

        // Pose of moved manipulator, in camera coordinate
        var camTr = _cameraSensor.Camera.transform;
        var rot = Quaternion.Inverse(camTr.rotation) * activeHand.rotation;
        var pos = camTr.InverseTransformPoint(activeHand.position);
        var poseString = $"{rot.ToString("F4")},{pos.ToString("F4")}";
        poseString = poseString.Replace("(", "").Replace(")", "").Replace(", ", "/");
        actionEffect += poseString;

        // Some invariants re. provided string label key
        if (labelKey == "SA")
            // Pick-up target is a non-atomic subassembly
            Assert.IsTrue(targetEnt.closestChildren.Count > 1);
        else
            // Pick-up target is an atomic part, where labelKey value of "GT" would
            // act as 'skeleton key' that would always fetch ground-truth pose
            Assert.IsTrue(targetEnt.closestChildren.Count == 1);
        
        // Involved (atomic) EnvEntity poses after picking up
        var atomicPartCount = 0;
        var partPosesString = "";
        foreach (var ent in activeHand.GetComponentsInChildren<EnvEntity>())
        {
            if (!ent.isAtomic) continue;
            atomicPartCount++;

            // Unique identifier for tracking the part
            partPosesString += $",{ent.name}";

            // 3D pose of the part
            var partTr = ent.gameObject.transform;
            var partRot = Quaternion.Inverse(camTr.rotation) * partTr.rotation;
            var partPos = camTr.InverseTransformPoint(partTr.position);

            // Add random perturbations to the ground-truth pose value if the target
            // entity is an atomic part, where label key provided is not the skeleton
            // key "GT" and is invalid. This simulates inaccurate pose estimation due
            // to incorrect part type assumption.
            if (labelKey != "GT" && labelKey != "SA")
            {
                if (ent.licensedLabel != labelKey)
                {
                    partRot = Random.rotationUniform;
                    partPos += Random.onUnitSphere * 0.1f;
                }
            }

            var poseSerialized = $"{partRot.ToString("F4")},{partPos.ToString("F4")}";
            poseSerialized = poseSerialized.Replace("(", "").Replace(")", "").Replace(", ", "/");
            partPosesString += $",{poseSerialized}";
        }
        actionEffect += $",{atomicPartCount}{partPosesString})";

        return actionEffect;
    }

    private string Drop(bool fromLeft)
    {
        var directionString = fromLeft ? "left" : "right";
        
        // Action aftermath info to return; none as of now
        var actionEffect = $"# Effect: drop_{directionString}()";

        // Drop an object currently held in the specified hand onto the tabletop
        var activeHand = fromLeft ? leftHand : rightHand;

        // Get handle of object
        var heldObj = activeHand.transform.GetChild(0).gameObject;
        
        // Move target object above (y) some random x/z-position within the main
        // work area partition, 'release' by nullifying the hand's child status,
        float xPos, zPos;
        if (heldObj.name == "truck")
        {
            // Assuming the tabletop is pretty much clear now...
            xPos = 0f;
            zPos = _mainPartitionPosition.z + 0.14f;
        }
        else
        {
            xPos = _mainPartitionPosition.x + Random.Range(-0.15f, 0.15f);
            zPos = _mainPartitionPosition.z + Random.Range(0f, 0.14f);
        }
        var volume = GetBoundingVolume(heldObj);
        var yPos = _mainPartitionPosition.y + volume.extents.y + 0.08f;
        heldObj.transform.parent = null;
        heldObj.transform.position = new Vector3(xPos, yPos, zPos);

        // Re-enable physics & fast-forward physics simulation until the object rests
        // on the desktop
        var objRigidbody = heldObj.GetComponent<Rigidbody>();
        objRigidbody.isKinematic = false;
        objRigidbody.detectCollisions = true;
        Physics.simulationMode = SimulationMode.Script;
        for (var i = 0; i < 2000; i++)
            Physics.Simulate(Time.fixedDeltaTime);
        Physics.simulationMode = SimulationMode.FixedUpdate;

        return actionEffect;
    }

    private string Assemble(
        string productName, string leftPoint, string rightPoint, bool rightToLeft
    )
    {
        var directionString = rightToLeft ? "right_to_left" : "left_to_right";

        // Action aftermath info to return; resultant product name, involved contact points
        // resp. from left and right sides, poses (position, quaternion) of the manipulator
        // before and after movement, pose of target-side manipulator, poses of atomic parts
        // contained in the assembled product
        var actionEffect = $"# Effect: assemble_{directionString}(";
        actionEffect += $"{productName},{leftPoint},{rightPoint},";

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
            for (var i = 0; i < candidateEnt.gameObject.transform.childCount; i++)
            {
                var candidatePt = candidateEnt.gameObject.transform.GetChild(i);
                if (candidatePt.gameObject.name != srcPointName) continue;
                srcPoint = candidatePt;
                break;
            }
            if (srcPoint is not null) break;
        }
        foreach (var candidateEnt in tgtHand.GetComponentsInChildren<EnvEntity>())
        {
            if (candidateEnt.gameObject.name != tgtAtomicPart) continue;
            for (var i = 0; i < candidateEnt.gameObject.transform.childCount; i++)
            {
                var candidatePt = candidateEnt.gameObject.transform.GetChild(i);
                if (candidatePt.gameObject.name != tgtPointName) continue;
                tgtPoint = candidatePt;
                break;
            }
            if (tgtPoint is not null) break;
        }
        Assert.IsNotNull(srcPoint); Assert.IsNotNull(tgtPoint);

        // Pose of moved manipulator before movement, in camera coordinate
        // (Note to self: transform.InverseTransformPoint method returns properly transformed
        // coordinate inv(R_tr)*(-t_tr+t_target). Well duh.)
        var camTr = _cameraSensor.Camera.transform;
        var rotBefore = Quaternion.Inverse(camTr.rotation) * srcHand.rotation;
        var posBefore = camTr.InverseTransformPoint(srcHand.position);
        var beforeString = $"{rotBefore.ToString("F4")},{posBefore.ToString("F4")}";
        beforeString = beforeString.Replace("(", "").Replace(")", "").Replace(", ", "/");

        // Get relative pose (position & rotation) from source to target points, then
        // move source hand to target pose (rotation first, translation next)
        var relativeRotation = tgtPoint.rotation * Quaternion.Inverse(srcPoint.rotation);
        srcHand.rotation = relativeRotation * srcHand.rotation;
        var relativePosition = tgtPoint.position - srcPoint.position;
        srcHand.position += relativePosition;

        // Pose of moved manipulator after movement, in camera coordinate
        var rotAfter = Quaternion.Inverse(camTr.rotation) * srcHand.rotation;
        var posAfter = camTr.InverseTransformPoint(srcHand.position);
        var afterString = $"{rotAfter.ToString("F4")},{posAfter.ToString("F4")}";
        afterString = afterString.Replace("(", "").Replace(")", "").Replace(", ", "/");

        // Pose of target side manipulator, in camera coordinate
        var rotTarget = Quaternion.Inverse(camTr.rotation) * tgtHand.rotation;
        var posTarget = camTr.InverseTransformPoint(tgtHand.position);
        var targetString = $"{rotTarget.ToString("F4")},{posTarget.ToString("F4")}";
        targetString = targetString.Replace("(", "").Replace(")", "").Replace(", ", "/");

        actionEffect += $"{beforeString},{afterString},{targetString}";

        // Handles to subassembly objects held in source & target hands
        var srcHeld = srcHand.transform.GetChild(0).gameObject;
        var tgtHeld = tgtHand.transform.GetChild(0).gameObject;

        // Obtain pre-assembly bounding volume for later repositioning of children parts
        var beforeVolume = GetBoundingVolume(tgtHeld);

        // Merge the two subassemblies, 'releasing' from source hand by reassigning
        // parent transforms of parts in the source subassembly
        for (var i = srcHeld.transform.childCount-1; i > -1; i--)
            srcHeld.transform.GetChild(i).parent = tgtHeld.transform;

        // Obtain post-assembly bounding volume for later repositioning of children parts
        var afterVolume = GetBoundingVolume(tgtHeld);

        // Finish merging by destroying source subassembly and renaming target subassembly
        // with the provided string name
        srcHeld.GetComponent<EnvEntity>().enabled = false;
        Destroy(srcHeld);
        tgtHeld.name = productName;

        // Update children of resulting subassembly
        tgtHeld.GetComponent<EnvEntity>().UpdateClosestChildren();

        // Repositioning children prefabs in merged subassembly so the center of the
        // bounding volume becomes origin of the local space
        for (var i = 0; i < tgtHand.transform.childCount; i++)
            tgtHand.transform.GetChild(i).position += beforeVolume.center - afterVolume.center;

        // Move back the source hand to the original pose
        srcHand.localPosition = rightToLeft ? rightOriginalPosition : leftOriginalPosition;
        srcHand.localEulerAngles = Vector3.zero;

        // Involved (atomic) EnvEntity poses on target side after assembly
        var atomicPartCount = 0;
        var partPosesString = "";
        foreach (var ent in tgtHand.GetComponentsInChildren<EnvEntity>())
        {
            if (!ent.isAtomic) continue;
            atomicPartCount++;

            // Unique identifier for tracking the part
            partPosesString += $",{ent.name}";

            // 3D pose of the part
            var partTr = ent.gameObject.transform;
            var partRot = Quaternion.Inverse(camTr.rotation) * partTr.rotation;
            var partPos = camTr.InverseTransformPoint(partTr.position);

            var poseSerialized = $"{partRot.ToString("F4")},{partPos.ToString("F4")}";
            poseSerialized = poseSerialized.Replace("(", "").Replace(")", "").Replace(", ", "/");
            partPosesString += $",{poseSerialized}";
        }
        actionEffect += $",{atomicPartCount}{partPosesString})";

        return actionEffect;
    }

    private string Assemble(
        string productName, Quaternion targetRot, Vector3 targetPos, bool rightToLeft
    )
    {
        var directionString = rightToLeft ? "right_to_left" : "left_to_right";

        // Action aftermath info to return; pose of target-side manipulator, up to
        // three closest contact point pairs and their pose (rotation and position)
        // differences, poses of atomic parts contained in the assembled product
        var actionEffect = $"# Effect: assemble_{directionString}(";

        // Source & target hands appropriately determined by `rightToLeft` parameter
        var srcHand = rightToLeft ? rightHand : leftHand;
        var tgtHand = rightToLeft ? leftHand : rightHand;

        // Desired rotation and position of manipulator to be moved are in the camera
        // coordinate; cast to values in global coordinate and set manipulator pose
        // accordingly
        var camTr = _cameraSensor.Camera.transform;
        srcHand.rotation = camTr.rotation * targetRot;
        srcHand.position = camTr.TransformPoint(targetPos);

        // Handles to subassembly objects held in source & target hands+
        var srcHeld = srcHand.transform.GetChild(0).gameObject;
        var tgtHeld = tgtHand.transform.GetChild(0).gameObject;

        // Consider all possible bipartite matching of contact points included in
        // each manipulator, compute rotation and position differences
        var cpsSrc = srcHeld.GetComponent<EnvEntity>().closestChildren
            .Select(ent => ent.gameObject.transform)
            .SelectMany(tr => tr.Cast<Transform>().ToList())
            .Where(obj => obj.name.StartsWith("cp_")).ToList();
        var cpsTgt = tgtHeld.GetComponent<EnvEntity>().closestChildren
            .Select(ent => ent.gameObject.transform)
            .SelectMany(tr => tr.Cast<Transform>().ToList())
            .Where(obj => obj.name.StartsWith("cp_")).ToList();
        var cpPairs = new List<(GameObject, GameObject, float, float)>();
        foreach (var cpTrSrc in cpsSrc)
        foreach (var cpTrTgt in cpsTgt)
        {
            var cpSrc = cpTrSrc.gameObject;
            var cpTgt = cpTrTgt.gameObject;
            var positionDiff = Vector3.Distance(cpTrSrc.position, cpTrTgt.position);
            var positionMatch = Mathf.Exp(
                -Mathf.Pow(positionDiff, 2) / (2 * Mathf.Pow(0.005f, 2))
            );      // Use Gaussian kernel with sigma of 0.01 to compute position match score,
                    // requiring precise position match in effect
            var rotationMatch = Math.Abs(Quaternion.Dot(cpTrSrc.rotation, cpTrTgt.rotation));
            cpPairs.Add((cpSrc, cpTgt, positionMatch, rotationMatch));
        }

        // 'Snap' by contact point pair with the smallest pose difference, equating
        // their poses
        cpPairs = cpPairs
            .OrderByDescending(pair => pair.Item3 + pair.Item4).ToList();
        var bestPair = cpPairs.First();
        var bestCpSrc = bestPair.Item1.transform;
        var bestCpTgt = bestPair.Item2.transform;

        // Get relative pose (position & rotation) from source to target points, then
        // further adjust manipulator pose to snap
        var relativeRotation = bestCpTgt.rotation * Quaternion.Inverse(bestCpSrc.rotation);
        srcHand.rotation = relativeRotation * srcHand.rotation;
        var relativePosition = bestCpTgt.position - bestCpSrc.position;
        srcHand.position += relativePosition;

        // Obtain pre-assembly bounding volume for later repositioning of children parts
        var beforeVolume = GetBoundingVolume(tgtHeld);

        // Merge the two subassemblies, 'releasing' from source hand by reassigning
        // parent transforms of parts in the source subassembly
        for (var i = srcHeld.transform.childCount-1; i > -1; i--)
            srcHeld.transform.GetChild(i).parent = tgtHeld.transform;

        // Obtain post-assembly bounding volume for later repositioning of children parts
        var afterVolume = GetBoundingVolume(tgtHeld);

        // Finish merging by destroying source subassembly and renaming target subassembly
        // with the provided string name
        srcHeld.GetComponent<EnvEntity>().enabled = false;
        Destroy(srcHeld);
        tgtHeld.name = productName;

        // Update children of resulting subassembly
        tgtHeld.GetComponent<EnvEntity>().UpdateClosestChildren();

        // Repositioning children prefabs in merged subassembly so the center of the
        // bounding volume becomes origin of the local space
        for (var i = 0; i < tgtHand.transform.childCount; i++)
            tgtHand.transform.GetChild(i).position += beforeVolume.center - afterVolume.center;

        // Move back the source hand to the original pose
        srcHand.localPosition = rightToLeft ? rightOriginalPosition : leftOriginalPosition;
        srcHand.localEulerAngles = Vector3.zero;

        // Pose of target side manipulator, in camera coordinate
        var rot = Quaternion.Inverse(camTr.rotation) * tgtHand.rotation;
        var pos = camTr.InverseTransformPoint(tgtHand.position);
        var poseString = $"{rot.ToString("F4")},{pos.ToString("F4")}";
        poseString = poseString.Replace("(", "").Replace(")", "").Replace(", ", "/");
        actionEffect += $"{poseString}";

        // Report up to three contact point pairs with the closest distances 
        var pairCount = 0;
        var matchStrings = new List<string>();
        foreach (var (cpSrc, cpTgt, positionMatch, rotationMatch) in cpPairs)
        {
            if (pairCount < 3)
                pairCount++;
            else
                break;

            var partTypeSrc = cpSrc.transform.parent.gameObject;
            var partTypeTgt = cpTgt.transform.parent.gameObject;
            var srcString = $"{partTypeSrc.name}/{cpSrc.name.Replace("cp_", "")}";
            var tgtString = $"{partTypeTgt.name}/{cpTgt.name.Replace("cp_", "")}";
            var posMatchString = positionMatch.ToString("F4");
            var rotMatchString = rotationMatch.ToString("F4");
            var matchString = $"{srcString},{tgtString},{posMatchString},{rotMatchString}";
            matchStrings.Add(matchString);
        }

        actionEffect += $",{pairCount}";
        foreach (var matchString in matchStrings) actionEffect += $",{matchString}";

        // Involved (atomic) EnvEntity poses on target side after assembly
        var atomicPartCount = 0;
        var partPosesString = "";
        foreach (var ent in tgtHand.GetComponentsInChildren<EnvEntity>())
        {
            if (!ent.isAtomic) continue;
            atomicPartCount++;

            // Unique identifier for tracking the part
            partPosesString += $",{ent.name}";

            // 3D pose of the part
            var partTr = ent.gameObject.transform;
            var partRot = Quaternion.Inverse(camTr.rotation) * partTr.rotation;
            var partPos = camTr.InverseTransformPoint(partTr.position);

            var poseSerialized = $"{partRot.ToString("F4")},{partPos.ToString("F4")}";
            poseSerialized = poseSerialized.Replace("(", "").Replace(")", "").Replace(", ", "/");
            partPosesString += $",{poseSerialized}";
        }
        actionEffect += $",{atomicPartCount}{partPosesString})";

        return actionEffect;
    }

    private string Inspect(int viewIndex, string inspectedObjName, bool onLeft)
    {
        // Move the specified hand to 'observation' position, then rotate according to the
        // specified viewing angle index. Index value of 24 indicates end of inspection,
        // bring the hand back to the original position.
        var directionString = onLeft ? "left" : "right";

        // Action aftermath info to return; 3d pose of the object being inspected
        // in camera coordinate, RLE encoding of current entity mask
        var actionEffect = $"# Effect: inspect_{directionString}(";

        var activeHand = onLeft ? leftHand : rightHand;

        var heldObj = activeHand.transform.GetChild(0).gameObject;
        Assert.IsTrue(heldObj.name == inspectedObjName);

        var distance = Vector3.Distance(
            relativeViewCenter.position, relativeViewPoint.position
        );

        if (viewIndex == 0)
        {
            // Need to do at the beginning of each inspection sequence
            inspectOriginalRotation = relativeViewCenter.rotation;
            activeHand.position = relativeViewCenter.position;
        }

        if (viewIndex < 24)
        {
            // Turn hand orientation to each direction where the imaginary viewer is supposed to be
            if (viewIndex % 8 == 0)
            {
                // Adjust 'viewing height'
                var rx = (viewIndex / 8) switch
                {
                    0 => _cameraSensor.Camera.transform.eulerAngles.x - 50f,
                    1 => _cameraSensor.Camera.transform.eulerAngles.x,
                    2 => _cameraSensor.Camera.transform.eulerAngles.x + 50f,
                    _ => relativeViewCenter.eulerAngles.x
                };
                relativeViewCenter.eulerAngles = new Vector3(rx, 0f, 0f);
            }
            else
                if (viewIndex % 2 == 0)
                    relativeViewCenter.Rotate(Vector3.up, 70f, Space.Self);
                else
                    relativeViewCenter.Rotate(Vector3.up, 20f, Space.Self);

            activeHand.LookAt(relativeViewPoint, relativeViewCenter.up);
        }

        if (viewIndex == 24)
        {
            // Back to default poses at the end of inspection
            activeHand.localPosition = onLeft ? leftOriginalPosition : rightOriginalPosition;
            activeHand.localEulerAngles = Vector3.zero;
            relativeViewPoint.localPosition = Vector3.forward * distance;
            relativeViewCenter.rotation = inspectOriginalRotation;
        }

        // 3d pose of object in camera coordinate needed to adjust for ground-truth
        // poses passed later
        var camTr = _cameraSensor.Camera.transform;
        var partTr = heldObj.transform.GetChild(0);
        var rot = Quaternion.Inverse(camTr.rotation) * partTr.rotation;
        var pos = camTr.InverseTransformPoint(partTr.position);
        var poseString = $"{rot.ToString("F4")},{pos.ToString("F4")}";
        poseString = poseString.Replace("(", "").Replace(")", "").Replace(", ", "/");
        actionEffect += $"{poseString},";

        // Sending current segmentation mask (RLE-encoded) of the EnvEntity as action effect
        // as well
        var entMaskRle = MessageSideChannel
            .RleEncode(GetSensorMask(heldObj.GetComponent<EnvEntity>()))
            .Select(f => f.ToString()).ToArray();
        var maskString = String.Join("/", entMaskRle);
        actionEffect += $"{maskString})";
        
        return actionEffect;
    }

    private string Disassemble(
        string leftName, string rightName, List<string> takeawayParts, bool fromLeft
    )
    {
        var directionString = fromLeft ? "left" : "right";

        // Action aftermath info to return; poses (position, quaternion) of both
        // manipulators after disassembly, poses of atomic parts contained
        // in each disassembled subassembly on each side 
        var actionEffect = $"# Effect: disassemble_{directionString}(";

        // Drop an object currently held in the specified hand onto the tabletop
        var activeHand = fromLeft ? leftHand : rightHand;
        var emptyHand = fromLeft ? rightHand : leftHand;

        // Get handle of object to be disassembled; update name
        var heldObj = activeHand.transform.GetChild(0).gameObject;
        heldObj.name = fromLeft ? leftName : rightName;

        // Create a new subassembly gameObject to hold the parts to be taken away
        var newObj = new GameObject(
            fromLeft ? rightName : leftName,
            typeof(EnvEntity), typeof(Rigidbody), typeof(Labeling)
        )
        {
            // To be held by the currently empty hand
            transform =
            {
                parent = emptyHand.transform,
                position = emptyHand.transform.position
            }
        };
        var newObjRigidbody = newObj.GetComponent<Rigidbody>();
        newObjRigidbody.isKinematic = true;
        newObjRigidbody.detectCollisions = false;

        // Disassemble the subassembly by reassigning parent transforms of parts to
        // take away from it
        for (var i = heldObj.transform.childCount - 1; i > -1; i--)
        {
            var part = heldObj.transform.GetChild(i).gameObject;
            if (takeawayParts.Contains(part.name))
                part.transform.parent = newObj.transform;
        }

        // Re-centering both subassemblies
        var leftObj = fromLeft ? heldObj : newObj;
        var rightObj = fromLeft ? newObj : heldObj;
        var leftVolume = GetBoundingVolume(leftObj);
        
        var rightVolume = GetBoundingVolume(rightObj);
        foreach (Transform partTr in leftObj.transform)
            partTr.position += leftHand.position - leftVolume.center;
        foreach (Transform partTr in rightObj.transform)
            partTr.position += rightHand.position - rightVolume.center;

        // Update children of both subassemblies
        leftObj.GetComponent<EnvEntity>().UpdateClosestChildren();
        rightObj.GetComponent<EnvEntity>().UpdateClosestChildren();

        // Pose of both manipulators, in camera coordinate
        var camTr = _cameraSensor.Camera.transform;
        var leftRot = Quaternion.Inverse(camTr.rotation) * leftHand.rotation;
        var leftPos = camTr.InverseTransformPoint(leftHand.position);
        var leftPoseString = $"{leftRot.ToString("F4")},{leftPos.ToString("F4")}";
        leftPoseString = leftPoseString.Replace("(", "").Replace(")", "").Replace(", ", "/");
        var rightRot = Quaternion.Inverse(camTr.rotation) * rightHand.rotation;
        var rightPos = camTr.InverseTransformPoint(rightHand.position);
        var rightPoseString = $"{rightRot.ToString("F4")},{rightPos.ToString("F4")}";
        rightPoseString = rightPoseString.Replace("(", "").Replace(")", "").Replace(", ", "/");
        actionEffect += $"{leftPoseString},{rightPoseString}";

        // Involved (atomic) EnvEntity poses on target side after assembly
        var bothObjs = new[] { leftObj.transform, rightObj.transform };
        foreach (var objTr in bothObjs)
        {
            var atomicPartCount = 0;
            var partPosesString = "";
            foreach (Transform partTr in objTr)
            {
                atomicPartCount++;

                // Unique identifier for tracking the part
                var ent = partTr.gameObject.GetComponent<EnvEntity>();
                partPosesString += $",{ent.name}";

                // 3D pose of the part
                var partRot = Quaternion.Inverse(camTr.rotation) * partTr.rotation;
                var partPos = camTr.InverseTransformPoint(partTr.position);

                var poseSerialized = $"{partRot.ToString("F4")},{partPos.ToString("F4")}";
                poseSerialized = poseSerialized.Replace("(", "").Replace(")", "").Replace(", ", "/");
                partPosesString += $",{poseSerialized}";
            }
            actionEffect += $",{atomicPartCount}{partPosesString}";
        }
        actionEffect += ")";

        return actionEffect;
    }

    protected static Bounds GetBoundingVolume(GameObject subassembly)
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

    // To be overridden in children classes; for packing & communicating information re.
    // ordering of concept subtypes as specified in current Unity scene
    protected virtual List<(string, List<string>)> SubtypeOrderings()
    {
        return new List<(string, List<string>)>();
    }

    private static float ContainsColor(Color32 color, Color32[] colorSet)
    {
        // Test if input color is contained in an array of colors
        return colorSet.Any(c => c.r == color.r && c.g == color.g && c.b == color.b) ? 1f : 0f;
    }

    private static float[] ColorsToMask(Color32[] segMapBuffer, Color32[] entities)
    {
        // Convert set of Color32 values to binary mask based on the segmentation image buffer
        var binaryMask = segMapBuffer
            .Select(c => ContainsColor(c, entities)).ToArray();

        return binaryMask;
    }

    private float[] MaskCoordinateSwitch(float[] sourceMask, bool screenToSensor)
    {
        var targetDisplay = _cameraSensor.Camera.targetDisplay;
        var screenWidth = Display.displays[targetDisplay].renderingWidth;
        var screenHeight = Display.displays[targetDisplay].renderingHeight;
        var sensorWidth = _cameraSensor.Width;
        var sensorHeight = _cameraSensor.Height;

        int sourceWidth, sourceHeight, targetWidth, targetHeight;
        if (screenToSensor)
        {
            sourceWidth = screenWidth; sourceHeight = screenHeight;
            targetWidth = sensorWidth; targetHeight = sensorHeight;
        }
        else
        {
            sourceWidth = sensorWidth; sourceHeight = sensorHeight;
            targetWidth = screenWidth; targetHeight = screenHeight;
        }

        // Texture2D representation of the provided mask to be manipulated, in the
        // screen coordinate
        var sourceTexture = new Texture2D(sourceWidth, sourceHeight);
        sourceTexture.SetPixels(
            sourceMask.Select(v => new Color(1f, 1f, 1f, v)).ToArray()
        );

        // To target coordinate, rescale by height ratio
        // (For some reason it's freakishly inconvenient to resize images in Unity)
        var heightRatio = (float) targetHeight / sourceHeight;
        var newWidth = (int) (sourceWidth * heightRatio);
        var rescaledSourceTexture = new Texture2D(newWidth, targetHeight);
        for (var i = 0; i < newWidth; i++)
        {
            var u = (float) i / newWidth;
            for (var j = 0; j < targetHeight; j++)
            {
                var v = (float) j / targetHeight;
                var interpolatedPixel = sourceTexture.GetPixelBilinear(u, v);
                rescaledSourceTexture.SetPixel(i, targetHeight-j, interpolatedPixel);
                    // Flip y-axis
            }
        }
        var rescaledMask = rescaledSourceTexture.GetPixels();

        // X-axis offset for copying over mask data
        var xOffset = (newWidth - targetWidth) / 2;

        // Read values from the rescaled sourceTexture, row by row, to obtain the
        // final mask transformation
        var targetMask = new float[targetWidth * targetHeight];
        for (var j = 0; j < targetHeight; j++)
        {
            int sourceStart, sourceEnd, targetStart;
            if (xOffset > 0)
            {
                // Source is 'wider' in aspect ratio, read with the x-axis offset
                sourceStart = j * newWidth + xOffset;
                sourceEnd = sourceStart + targetWidth;
                targetStart = j * targetWidth;
            }
            else
            {
                // Source is 'narrower' in aspect ratio, write with the x-axis offset
                sourceStart = j * newWidth;
                sourceEnd = sourceStart + newWidth;
                targetStart = j * targetWidth - xOffset;
            }

            for (var i = 0; i < sourceEnd-sourceStart; i++)
                targetMask[i+targetStart] = rescaledMask[i+sourceStart].a;
        }

        // Free them
        Destroy(sourceTexture);
        Destroy(rescaledSourceTexture);

        return targetMask;
    }

    private float[] GetSensorMask(EnvEntity ent)
    {
        float[] entMask;
        if (ent.isBogus)
        {
            // Bogus entity, mask directly stores the bitmap
            entMask = ent.masks[_cameraSensor.Camera.targetDisplay]
                .Select(c => c.a > 0f ? 1f : 0f).ToArray();
        }
        else
        {
            // Non-bogus entity, mask stores set of matching colors, which need to be
            // translated to bitmap

            // Retrieve referenced EnvEntity and fetch segmentation mask in absolute scale
            // w.r.t. this agent's camera's target display screen
            var maskColors = ent.masks[_cameraSensor.Camera.targetDisplay];
            var segMapBuffer = EnvEntity.annotationStorage.segMap;
            entMask = ColorsToMask(segMapBuffer, maskColors);
            entMask = MaskCoordinateSwitch(entMask, true);
        }

        return entMask;
    }
}
