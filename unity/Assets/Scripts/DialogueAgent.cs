using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Perception.GroundTruth;
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
    public readonly Queue<(string, string, Dictionary<(int, int), EntityRef>)> outgoingMsgBuffer = new();

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

    // A bit of 'cheating' for stabilizing pose estimation results, as physical
    // simulations may perturb rotation after dropping objects. Cache localRotations
    // of held objects when dropping, and restore when picking them
    // back up
    private Dictionary<string, Quaternion> _rotationCache;
    
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
        _rotationCache = new Dictionary<string, Quaternion>();
        _nextTimeToAct = Time.time;

        leftOriginalPosition = leftHand.localPosition;
        rightOriginalPosition = rightHand.localPosition;

        dialogueChannel.dialogueParticipants.Add(this);
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
                foreach (var (range, entMask) in incomingMessage.demonstrativeReferences)
                    demRefs[range] = new EntityRef(entMask);

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
                _cameraSensor.Camera.transform.eulerAngles = new Vector3(55f, 0f, 0f);

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
        for (var i=0; i < 5; i++)
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
        var messagesToUtter = new List<(string, string, Dictionary<(int, int), EntityRef>)>();
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
                        );
                    foreach (var ent in topEntities)
                        gtMaskRequests.Enqueue(ent.gameObject.name);
                }
                else
                {
                    partStrings.Add(req);

                    var range = (stringPointer, stringPointer + req.Length);

                    // Find relevant EnvEntity and fetch mask
                    var foundEnt = EnvEntity.FindByObjectPath("/" + req);
                    responseMasks[range] = new EntityRef(GetSensorMask(foundEnt));

                    stringPointer += req.Length;
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
                var demRefsResolved = new Dictionary<(int, int), float[]>();
                var targetDisplay = _cameraSensor.Camera.targetDisplay;

                foreach (var (range, demRef) in demRefs)
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
                    demRefsResolved[range] = GetSensorMask(ent);
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
        var referencedEntities = new Dictionary<(int, int), EnvEntity>();

        // Coroutine that executes specified physical action
        switch (actionType)
        {
            case 2:
            case 3:
                // PickUpLeft/Right action, parameter: {target object}, where target
                // may be designated by GameObject path or segmentation mask
                var withLeft = actionType % 2 == 0;
                var pickUpParam1 = actionParameterBuffer.Dequeue();
                var targetName = pickUpParam1.Item1.Replace("str|", "");
                EnvEntity targetEnt;
                if (targetName == "@DemRef")
                {
                    var prmMask = pickUpParam1.Item2.maskRef;
                    var screenMask = MaskCoordinateSwitch(prmMask, false);
                    var targetDisplay = _cameraSensor.Camera.targetDisplay;
                    targetEnt = EnvEntity.FindByMask(screenMask, targetDisplay, true);
                    Assert.IsFalse(targetEnt.isBogus);
                }
                else
                    targetEnt = EnvEntity.FindByObjectPath($"/{targetName}");
                (actionEffect, referencedEntities) = PickUp(targetEnt, withLeft);
                break;
            case 4:
            case 5:
                // DropLeft/Right action, parameter: ()
                var fromLeft = actionType % 2 == 0;
                Drop(fromLeft);
                break;
            case 6:
            case 7:
                // AssembleRtoL/LtoR action, two possible parameter signatures:
                //  1) {resultant subassembly name, object/contact L, object/contact R}
                //  2) {resultant subassembly name, manipulator transform}
                var rightToLeft = actionType % 2 == 0;
                var assembleParam1 = actionParameterBuffer.Dequeue();
                var productName = assembleParam1.Item1.Replace("str|", "");
                
                var assembleParam2 = actionParameterBuffer.Dequeue();
                if (assembleParam2.Item1.StartsWith("str|"))
                {
                    var leftPoint = assembleParam2.Item1.Replace("str|", "");
                    var assembleParam3 = actionParameterBuffer.Dequeue();
                    var rightPoint = assembleParam3.Item1.Replace("str|", "");
                    (actionEffect, referencedEntities) = Assemble(
                        productName, leftPoint, rightPoint, rightToLeft
                    );
                }
                else
                {
                    Assert.IsTrue(assembleParam2.Item1.StartsWith("floats|"));
                    var handTransform = assembleParam2.Item1
                        .Replace("floats|", "").Split("/").Select(float.Parse).ToList();
                }
                break;
            case 8:
            case 9:
                // InspectLeft/Right action, parameter: {view angle index}
                var onLeft = actionType % 2 == 0;
                var inspectParam1 = actionParameterBuffer.Dequeue();
                var inspectParam2 = actionParameterBuffer.Dequeue();
                var inspectedObjName = inspectParam1.Item1.Replace("str|", "");
                var viewAngleIndex = Convert.ToInt32(inspectParam2.Item1.Replace("int|", ""));
                Inspect(viewAngleIndex, inspectedObjName, onLeft);
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
            var demRefsSelf = new Dictionary<(int, int), EntityRef>();
            var demRefsBroadcast = new Dictionary<(int, int), float[]>();
            // Wait for masks to be updated if needed
            if (referencedEntities is not null && referencedEntities.Count > 0)
            {
                foreach (var (range, ent) in referencedEntities)
                {
                    demRefsSelf[range] = new EntityRef(GetSensorMask(ent));
                    demRefsBroadcast[range] = demRefsSelf[range].maskRef;
                }
            }

            // To self and broadcast to others
            if (this is StudentAgent)
            {
                // Student: Recording action effect and send to self's backend
                backendMsgChannel.SendMessageToBackend(
                    dialogueParticipantID, actionEffect, demRefsSelf
                );
                // Send a null message as content-less ping, so that student will
                // request next step's decision from backend
                incomingMsgBuffer.Enqueue(null);
            }
            // Student & Teacher: Broadcast action effect to student
            dialogueChannel.CommitUtterance(
                dialogueParticipantID, actionEffect, demRefsBroadcast
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

    private (string, Dictionary<(int, int), EnvEntity>) PickUp(EnvEntity targetEnt, bool withLeft)
    {
        var directionString = withLeft ? "left" : "right";

        // Action aftermath info; name of object picked up, pose of manipulator, masks
        // of atomic parts contained in the object picked up
        var actionEffect = $"# Effect: pick_up_{directionString}(";
        var demRefs = new Dictionary<(int, int), EnvEntity>();

        // Pick up a target object on the tabletop with the specified hand
        var activeHand = withLeft ? leftHand : rightHand;

        // Move target object to hand position, reassign hand as the parent,
        // disable physics interaction
        var targetObj = targetEnt.gameObject;
        targetObj.transform.parent = activeHand.transform;
        targetObj.transform.localPosition = Vector3.zero;
        if (_rotationCache.TryGetValue(targetObj.name, out var cachedRotation))
            targetObj.transform.localRotation = cachedRotation;
        var objRigidbody = targetObj.GetComponent<Rigidbody>();
        objRigidbody.isKinematic = true;
        objRigidbody.detectCollisions = false;

        // Pose of moved manipulator, in camera coordinate
        var camTr = _cameraSensor.Camera.transform;
        var rot = Quaternion.Inverse(camTr.rotation) * activeHand.rotation;
        var pos = camTr.InverseTransformPoint(activeHand.position);
        var poseString = $"{rot.ToString("F4")},{pos.ToString("F4")}";
        poseString = poseString.Replace("(", "").Replace(")", "").Replace(", ", "/");
        actionEffect += poseString;

        // Report Unity-side gameObject name of the picked up entity
        actionEffect += "," + targetObj.name;

        // Involved (atomic) EnvEntity masks after picking up
        var offset = actionEffect.Length;
        foreach (var ent in activeHand.GetComponentsInChildren<EnvEntity>())
        {
            if (!ent.isAtomic) continue;

            var fragment = $",{ent.gameObject.name}";
            var range = (offset + 1, offset + fragment.Length);
            actionEffect += fragment;
            demRefs[range] = ent;
            offset += fragment.Length;
        }
        actionEffect += ")";

        return (actionEffect, demRefs);
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

        // Cache the current local rotation so that it can be restored later
        _rotationCache[heldObj.name] = heldObj.transform.localRotation;
        
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
            zPos = _mainPartitionPosition.z + Random.Range(-0.08f, 0.12f);
        }
        var volume = GetBoundingVolume(heldObj);
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

    private (string, Dictionary<(int, int), EnvEntity>) Assemble(
        string productName, string leftPoint, string rightPoint, bool rightToLeft
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

        // Queue manipulator pose change information to message to backend; pose (position,
        // quaternion) before, pose after
        var directionString = rightToLeft ? "right_to_left" : "left_to_right";
        var actionEffect = $"# Effect: assemble_{directionString}({beforeString},{afterString})";

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
        var beforeVolume = GetBoundingVolume(tgtHeld);

        // Merge the two subassemblies, 'releasing' from source hand by reassigning
        // parent transforms of parts in the source subassembly
        var childrenParts = new List<Transform>();
        foreach (Transform tr in srcHeld.transform) childrenParts.Add(tr);
        foreach (var tr in childrenParts) tr.parent = tgtHeld.transform;

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
        foreach (Transform child in tgtHeld.transform)
            child.position += beforeVolume.center - afterVolume.center;

        // Move back the source hand to the original pose
        srcHand.localPosition = rightToLeft ? rightOriginalPosition : leftOriginalPosition;
        srcHand.localEulerAngles = Vector3.zero;

        return (actionEffect, null);
    }

    private void Inspect(int viewIndex, string inspectedObjName, bool onLeft)
    {
        // Move the specified hand to 'observation' position, then rotate according to the
        // specified viewing angle index. Index value of 40 indicates end of inspection,
        // bring the hand back to the original position.
        var activeHand = onLeft ? leftHand : rightHand;

        GameObject heldObj = null;
        foreach (Transform tr in activeHand.transform)
        {
            heldObj = tr.gameObject;
            break;
        }
        Assert.IsNotNull(heldObj);
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

        if (viewIndex < 40)
        {
            // Turn hand orientation to each direction where the imaginary viewer is supposed to be
            if (viewIndex % 8 == 0)
            {
                // Adjust 'viewing height'
                var rx = (viewIndex / 8) switch
                {
                    0 => _cameraSensor.Camera.transform.eulerAngles.x - 70f,
                    1 => _cameraSensor.Camera.transform.eulerAngles.x - 50f,
                    2 => _cameraSensor.Camera.transform.eulerAngles.x,
                    3 => _cameraSensor.Camera.transform.eulerAngles.x + 50f,
                    4 => _cameraSensor.Camera.transform.eulerAngles.x + 70f,
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

        if (viewIndex == 40)
        {
            // Back to default poses at the end of inspection
            activeHand.localPosition = onLeft ? leftOriginalPosition : rightOriginalPosition;
            activeHand.localEulerAngles = Vector3.zero;
            relativeViewPoint.localPosition = Vector3.forward * distance;
            relativeViewCenter.rotation = inspectOriginalRotation;
        }
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
