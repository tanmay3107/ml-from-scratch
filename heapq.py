# top_k_retriever.py
import heapq

def get_top_k_scores(score_stream, k):
    """
    Tracks the top K highest scores from a streaming iterator using a Min-Heap.
    Time Complexity: O(N log K)
    Space Complexity: O(K)
    """
    # Initialize an empty list to act as our min-heap
    min_heap = []
    
    for score in score_stream:
        # If we haven't filled our top-K quota yet, push the score directly
        if len(min_heap) < k:
            heapq.heappush(min_heap, score)
        # If the heap is full, compare the current score to the smallest element in our top-K
        elif score > min_heap[0]:
            # Pop the smallest element and push the new, larger score
            heapq.heappushpop(min_heap, score)
            
    # The heap now contains the top K highest elements, sorted in ascending order
    # We sort it descending at the very end to present a clean ranked list
    return sorted(min_heap, reverse=True)


# --- Quick Test ---
if __name__ == "__main__":
    # Simulated stream of incoming cosmic similarity scores from an embedding database
    incoming_scores = [0.45, 0.88, 0.12, 0.94, 0.72, 0.65, 0.99, 0.81, 0.34]
    K = 3
    
    print(f"🌊 Processing stream of {len(incoming_scores)} elements...")
    top_k = get_top_k_scores(incoming_scores, K)
    
    print(f"🏆 Top {K} highest ranked scores: {top_k}")