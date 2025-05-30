using System;
using System.Collections.Generic;
using System.Linq;
using Unity.MLAgents.SideChannels;

public class MessageSideChannel : SideChannel
{
    // Associated agent in scene
    private readonly DialogueAgent _listeningAgent;

    public MessageSideChannel(string channelUuid, DialogueAgent agent)
    {
        ChannelId = new Guid(channelUuid);
        _listeningAgent = agent;
    }

    protected override void OnMessageReceived(IncomingMessage msg)
    {
        // Read until end of message
        var speaker = msg.ReadString();
        var utterance = msg.ReadString();

        // Retrieve any demonstrative references (map from substring indices to
        // segmentation mask image, having same dimension as scene image perceived
        // by sensor) until end of the current message is reached (signalled by -1)
        var demRefs = new Dictionary<(int, int), (EntityRef, bool)>();
        while (true)
        {
            var intMessage = msg.ReadInt32();
        
            if (intMessage == -1) break;        // End of dem. refs list
            
            var start = intMessage;
            var end = msg.ReadInt32();
            var resolveByMask = msg.ReadBoolean();

            EntityRef entRef;
            if (resolveByMask)
            {
                var rleMask = msg.ReadFloatList().ToArray();
                entRef = new EntityRef(RleDecode(rleMask));
            }
            else
                entRef = new EntityRef(msg.ReadString());

            var outputAsMask = msg.ReadBoolean();
            demRefs[(start, end)] = (entRef, outputAsMask);
        }

        // Handle system requests or action parameter specifications from backend
        if (speaker == "System")
        {
            // Handle any calibration image requests
            if (utterance.StartsWith("# Calibration image request: "))
            {
                var request = utterance.Replace("# Calibration image request: ", "");
                _listeningAgent.calibrationImageRequest = Convert.ToInt32(request);
            }

            // Handle any part subtypes ordering request
            if (utterance == "# Subtype orderings request")
                _listeningAgent.subtypeOrderingRequest = true;

            // Handle any ground-truth masks info requests
            if (utterance.StartsWith("# GT mask request: "))
            {
                var requests = utterance.Replace("# GT mask request: ", "");
                foreach (var req in requests.Split(", "))
                    _listeningAgent.gtMaskRequests.Enqueue(req);
            }

            // Receive and store action parameters; string parameters (which are not
            // handled by MLAgent Package) are sent via this channel
            if (utterance.StartsWith("# Action parameters: "))
            {
                var parameters = utterance.Replace("# Action parameters: ", "");
                if (parameters != "")
                {
                    var offset = 0;
                    foreach (var prmString in parameters.Split(", "))
                    {
                        var prmRef = prmString == "str|@DemRef" ?
                            demRefs[(offset+4, offset+11)].Item1 : null;
                        _listeningAgent.actionParameterBuffer.Enqueue((prmString, prmRef));
                        offset += prmString.Length + 2;
                    }
                }
            }
        }

        // Put processed message data into incoming buffer
        if (speaker == _listeningAgent.dialogueParticipantID)
            _listeningAgent.outgoingMsgBuffer.Enqueue((speaker, utterance, demRefs));
    }

    public void SendMessageToBackend(
        string speaker, string utterance, Dictionary<(int, int), EntityRef> optionalDemRefs = null
    )
    {
        // Create OutgoingMessage instance (using for dispose at the end)
        using var msgOut = new OutgoingMessage();

        // Write speaker info & utterance content as string
        msgOut.WriteString(speaker);
        msgOut.WriteString(utterance);

        // (If any) Encode demonstrative references as two consecutive ints (marking
        // start & end of corresponding demonstrative pronoun substring) and either
        // float[] ((soft) segmentation mask) or string (direct reference by string
        // name of EnvEntity)
        if (optionalDemRefs is not null)
        {
            foreach (var (range, demRef) in optionalDemRefs)
            {
                var (start, end) = range;
                msgOut.WriteInt32(start);
                msgOut.WriteInt32(end);
                switch (demRef.refType)
                {
                    case EntityRefType.Mask:
                        msgOut.WriteBoolean(true);
                        msgOut.WriteFloatList(RleEncode(demRef.maskRef));
                        break;
                    case EntityRefType.String:
                        msgOut.WriteBoolean(false);
                        msgOut.WriteString(demRef.stringRef);
                        break;
                    default:
                        // Shouldn't reach here but anyway
                        throw new Exception("Invalid reference data type?");
                }
            }
        }

        // Mark end of message segment
        msgOut.WriteInt32(-1);

        // Queue message to send
        QueueMessageToSend(msgOut);
    }

    public static float[] RleEncode(float[] rawMask)
    {
        // Encode raw binary mask into RLE format for message compression
        var rle = new List<float>();

        var zeroFlag = true;
        var run = 0f;
        foreach (var f in rawMask)
        {
            // Increment run length if value matches with current flag; push current
            // run length value to return array and flip sign
            if (zeroFlag)
            {
                if (f == 0f) run += 1;
                else
                {
                    rle.Add(run);
                    zeroFlag = false;
                    run = 1f;
                }
            }
            else
            {
                if (f > 0f) run += 1;
                else
                {
                    rle.Add(run);
                    zeroFlag = true;
                    run = 1f;
                }
            }
        }
        if (run > 0f) rle.Add(run);     // Flush last entry

        return rle.ToArray();
    }

    private static float[] RleDecode(float[] rleMask)
    {
        // Decode RLE to recover raw binary mask
        var totalLength = (int) rleMask.Sum();
        var raw = new float[totalLength];

        var zeroFlag = true;
        var cumulative = 0; 
        foreach (var f in rleMask)
        {
            // Get integer run length and update values 
            var runLength = (int) f;
            for (var i = cumulative; i < cumulative+runLength; i++)
                raw[i] = zeroFlag ? 0f : 1f;

            // Flip sign and update cumulative index
            zeroFlag = !zeroFlag;
            cumulative += runLength;
        }

        return raw;
    }
}
