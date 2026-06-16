def longest_unique_token_sequence(token_stream):
    """
    Finds the maximum length of a contiguous sequence of unique tokens.
    Uses the Sliding Window (Two-Pointer) pattern for O(N) performance.
    
    :param token_stream: List of string tokens or integers.
    :return: Integer representing the maximum window size.
    """
    token_index_map = {}  # Dictionary to remember the last seen index of a token
    max_length = 0
    left_pointer = 0      # The start of our sliding window
    
    for right_pointer, token in enumerate(token_stream):
        # If we have seen this token before AND it is inside our current window
        if token in token_index_map and token_index_map[token] >= left_pointer:
            # We hit a duplicate! Shrink the window by moving the left pointer
            # right past the previous occurrence of this token.
            left_pointer = token_index_map[token] + 1
            
        # Update the token's most recent index
        token_index_map[token] = right_pointer
        
        # Calculate the current window size and update max_length
        current_window_size = right_pointer - left_pointer + 1
        max_length = max(max_length, current_window_size)
        
    return max_length

# --- Quick Test ---
if __name__ == "__main__":
    # Simulated stream of incoming word tokens
    stream = ["the", "cat", "sat", "on", "the", "mat", "with", "the", "cat"]
    
    print(f"🌊 Processing token stream: {stream}")
    max_len = longest_unique_token_sequence(stream)
    
    print(f"🏆 Longest unique sequence length: {max_len}")
    # Expected output: 6 ("cat", "sat", "on", "the", "mat", "with")