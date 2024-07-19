using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Perception.GroundTruth;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;
using Random = UnityEngine.Random;

public class DialogueAgent : Agent
// Communication with dialogue UI and communication with Python backend may be
// decoupled later?
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
    public readonly Queue<string> actionParameterBuffer = new();

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

        dialogueChannel.dialogueParticipants.Add(this);
    }

    public override void OnEpisodeBegin()
    {
        // Say anything this agent has to say
        StartCoroutine(Utter());
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
                var rx = quot12 == 0 ? 15f : 30f;
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
                _cameraSensor.Camera.transform.eulerAngles = new Vector3(50f, 0f, 0f);

                // Turn off request flag
                calibrationImageRequest = -1;
            }

            // If any part subtype ordering info requests are pending, handle them here by
            // sending info to backend
            if (subtypeOrderingRequest)
            {
                var responseString = "Subtype orderings response: ";
                var orderingInfo = SubtypeOrderings();

                var responseSubstrings = new List<string>();
                foreach (var (supertype, subtypes) in orderingInfo)
                {
                    var substring = $"{supertype} - "; 
                    substring += string.Join(", ", subtypes.ToArray());
                    responseSubstrings.Add(substring);
                }
                responseString += string.Join(" // ", responseSubstrings.ToArray());

                var emptyDemRefs = new Dictionary<(int, int), EntityRef>();
                backendMsgChannel.SendMessageToBackend(
                    "System", responseString, emptyDemRefs
                );
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

    protected IEnumerator Utter()
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
        if (messagesToUtter.Count == 0)
        {
            _uttering = false;
            yield break;
        }

        // Check if any of the messages has non-empty demonstrative references and
        // thus segmentation masks need to be captured
        var masksNeeded = messagesToUtter
            .Select(m => m.Item3)
            .Any(rfs => rfs is not null && rfs.Count > 0);

        // If needed, synchronously wait until masks are updated and captured
        if (masksNeeded)
        {
            yield return StartCoroutine(CaptureAnnotations());

            // If any ground-truth mask requests are pending, handle them here by
            // sending info to backend
            if (gtMaskRequests.Count > 0)
            {
                var responseString = "GT mask response: ";
                var responseMasks = new Dictionary<(int, int), EntityRef>();
                var stringPointer = responseString.Length;

                var partStrings = new List<string>();
                while (gtMaskRequests.Count > 0)
                {
                    var req = gtMaskRequests.Dequeue();
                    partStrings.Add(req);

                    var range = (stringPointer, stringPointer+req.Length);
                    
                    // Find relevant EnvEntity and fetch mask
                    var foundEnt = FindObjectsByType<EnvEntity>(FindObjectsSortMode.None)
                        .FirstOrDefault(
                            ent =>
                            {
                                var parent = ent.gameObject.transform.parent;
                                var hasParent = parent is not null;
                                return hasParent && req == parent.gameObject.name;
                            }
                        );

                    if (foundEnt is null) throw new Exception("Invalid part type");
                    responseMasks[range] = new EntityRef(GetSensorMask(foundEnt));

                    stringPointer += req.Length;
                    if (gtMaskRequests.Count > 0) stringPointer += 2;   // Account for ", " delimiter
                }
                responseString += string.Join(", ", partStrings.ToArray());

                backendMsgChannel.SendMessageToBackend(
                    "System", responseString, responseMasks
                );
            }
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
                            // Shouldn't reach here but anyways
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
    protected IEnumerator Act(int actionType)
    {
        // If a coroutine invocation is still running, do not start another; else,
        // set flag
        if (_acting) yield break;
        _acting = true;

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
            case 8:
                // InspectLeftRight action, parameter: (view angle index)
                var inspectedObjName = actionParameterBuffer.Dequeue();
                var viewAngleIndex = Convert.ToInt32(actionParameterBuffer.Dequeue());
                var onLeft = actionType % 2 == 1;
                Inspect(viewAngleIndex, inspectedObjName, onLeft);
                break;
        }

        // All parameters consumed
        Assert.IsTrue(actionParameterBuffer.Count == 0);

        // Changes made to environment, Perception cameras need capture again
        EnvEntity.annotationStorage.annotationsUpToDate = false;

        // Waiting several more frames to ensure visual observations (containing action
        // post-conditions) are properly captured
        for (var i=0; i < 10; i++)
            yield return null;

        // Reset flag on exit
        _acting = false;
    }

    protected IEnumerator UtterThenAct(int actionType)
    {
        // Coroutine that first invokes Utter(), waits until it finishes, and then
        // execute specified physical action
        // ReSharper disable once Unity.PerformanceCriticalCodeInvocation
        yield return StartCoroutine(Utter());
        yield return StartCoroutine(Act(actionType));
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

        // Left & right manipulator pose before movement, in camera coordinate
        var camTr = _cameraSensor.Camera.transform;
        var posLeft = camTr.InverseTransformPoint(leftHand.position);
        var rotLeft = Quaternion.Inverse(camTr.rotation) * leftHand.rotation;
        var leftString = $"{posLeft.ToString()},{rotLeft.ToString()}";
        leftString = leftString.Replace("(", "").Replace(")", "").Replace(", ", "/");
        var posRight = camTr.InverseTransformPoint(rightHand.position);
        var rotRight = Quaternion.Inverse(camTr.rotation) * rightHand.rotation;
        var rightString = $"{posRight.ToString()},{rotRight.ToString()}";
        rightString = rightString.Replace("(", "").Replace(")", "").Replace(", ", "/");

        // Get relative pose (position & rotation) from source to target points, then
        // move source hand to target pose (rotation first, translation next)
        var relativeRotation = tgtPoint.rotation * Quaternion.Inverse(srcPoint.rotation);
        srcHand.rotation = relativeRotation * srcHand.rotation;
        var relativePosition = tgtPoint.position - srcPoint.position;
        srcHand.position += relativePosition;

        // Pose of moved manipulator after movement, in camera coordinate
        var posAfter = camTr.InverseTransformPoint(srcHand.position);
        var rotAfter = Quaternion.Inverse(camTr.rotation) * srcHand.rotation;
        var afterString = $"{posAfter.ToString()},{rotAfter.ToString()}";
        afterString = afterString.Replace("(", "").Replace(")", "").Replace(", ", "/");

        // Queue manipulator pose change information to message to backend; pose (position,
        // quaternion) before, pose after
        var directionString = rightToLeft ? "RightToLeft" : "LeftToRight";
        var actionEffect = $"# Effect: Assemble{directionString}({leftString},{rightString},{afterString})";
        dialogueChannel.CommitUtterance(dialogueParticipantID, actionEffect);

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

        // Repositioning children prefabs in merged subassembly so the center of the
        // bounding volume becomes origin of the local space
        foreach (Transform child in tgtHeld.transform)
            child.position += beforeVolume.center - afterVolume.center;

        // Move back the source hand to the original pose
        srcHand.localPosition = rightToLeft ? rightOriginalPosition : leftOriginalPosition;
        srcHand.localEulerAngles = Vector3.zero;
    }

    private void Inspect(int viewIndex, string inspectedObjName, bool onLeft)
    {
        // Move the specified hand to 'observation' position, then rotate according to the
        // specified viewing angle index. Index value of 16 indicates end of inspection,
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

        if (viewIndex < 16)
        {
            // Turn hand orientation to each direction where the imaginary viewer is supposed to be
            if (viewIndex % 4 == 0)
            {
                // Adjust 'viewing height' (0~3: upper-high, 4~7: upper-low, 8~11: lower-low, 12~15:
                // lower-high)
                var rx = (viewIndex / 4) switch
                {
                    0 => _cameraSensor.Camera.transform.eulerAngles.x - 45f,
                    1 => _cameraSensor.Camera.transform.eulerAngles.x - 22.5f,
                    2 => _cameraSensor.Camera.transform.eulerAngles.x + 22.5f,
                    3 => _cameraSensor.Camera.transform.eulerAngles.x + 45f,
                    _ => relativeViewCenter.eulerAngles.x
                };
                relativeViewCenter.eulerAngles = new Vector3(rx, 0f, 0f);
            }
            else
                relativeViewCenter.Rotate(Vector3.up, 90f, Space.Self);

            activeHand.LookAt(relativeViewPoint, relativeViewCenter.up);
        }

        if (viewIndex == 16)
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
