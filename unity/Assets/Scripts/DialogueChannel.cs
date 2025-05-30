using System.Collections.Generic;
using System.Collections.ObjectModel;
using UnityEngine;

[CreateAssetMenu]
public class DialogueChannel : ScriptableObject
{
    // Serves as inter-agent communication channel

    // List of dialogue participants
    public Dictionary<string, DialogueAgent> dialogueParticipants = new();

    public void CommitUtterance(
        string speaker, string inputString,
        Dictionary<(int, int), (float[], string)> optionalDemRefs = null
    )
    {
        // Create a new record containing the utterance data
        var inputRecord = CreateInstance<RecordData>();
        inputRecord.speaker = speaker;
        inputRecord.utterance = inputString;
        if (optionalDemRefs is null)
        {
            var empty = new Dictionary<(int, int), (float[], string)>();
            inputRecord.demonstrativeReferences =
                new ReadOnlyDictionary<(int, int), (float[], string)>(empty);
        }
        else
        {
            // Use provided demRefs, without refreshing current one in dialogue channel
            inputRecord.demonstrativeReferences =
                new ReadOnlyDictionary<(int, int), (float[], string)>(optionalDemRefs);
        }

        // Broadcast the record to all audience members
        if (speaker == "System") return;        // Don't broadcast System 'messages'
        foreach (var agt in dialogueParticipants.Values)
        {
            if (agt.dialogueParticipantID != speaker)
                agt.incomingMsgBuffer.Enqueue(inputRecord);
        }
    }
}