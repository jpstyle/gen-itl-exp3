# Taken from [project root]/tools/message_side_channel.py, exact same code
def rle_decode(rle_mask):
    # Decode RLE to recover raw binary mask
    total_length = int(sum(rle_mask))
    raw = [None] * total_length

    zero_flag = True
    cumulative = 0
    for f in rle_mask:
        # Get integer run length and update values
        run_length = int(f)
        for i in range(cumulative, cumulative+run_length):
            raw[i] = 0 if zero_flag else 1
        
        # Flip sign and update cumulative index
        zero_flag = not zero_flag
        cumulative += run_length

    return raw
