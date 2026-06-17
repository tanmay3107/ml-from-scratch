# topological_sort.py
from collections import defaultdict

class ComputationalGraph:
    def __init__(self):
        # Adjacency list mapping a node to the nodes it depends on
        self.graph = defaultdict(list)
        
    def add_dependency(self, node, dependency):
        """
        Defines a directed edge.
        Example: To compute 'Loss', we first need 'Output'.
        """
        self.graph[node].append(dependency)

    def topological_sort(self, start_node):
        """
        Executes a Depth-First Search (DFS) to find the correct execution order.
        Time Complexity: O(V + E) where V is vertices (tensors) and E is edges (operations).
        """
        visited = set()
        execution_order = []

        def dfs(current_node):
            # If we've already resolved this node, skip it
            if current_node in visited:
                return
                
            # Mark the node as visited
            visited.add(current_node)
            
            # Recursively dive into the dependencies FIRST
            for dependency in self.graph[current_node]:
                dfs(dependency)
                
            # Once all dependencies of the current node are resolved, 
            # we can safely add the current node to our execution path.
            execution_order.append(current_node)

        # Kick off the recursive search from the final node (e.g., the Loss function)
        dfs(start_node)
        
        return execution_order


# --- Quick Test ---
if __name__ == "__main__":
    print("🧠 Initializing Computational Graph (Autograd Simulator)...")
    autograd = ComputationalGraph()
    
    # Simulating the forward pass dependencies of a basic Neural Network
    # Format: add_dependency(Target, Requirement)
    autograd.add_dependency("Loss", "Predictions")
    autograd.add_dependency("Loss", "Ground Truth Labels")
    autograd.add_dependency("Predictions", "Hidden Layer 2")
    autograd.add_dependency("Hidden Layer 2", "Hidden Layer 1")
    autograd.add_dependency("Hidden Layer 2", "Weights 2")
    autograd.add_dependency("Hidden Layer 1", "Input Data")
    autograd.add_dependency("Hidden Layer 1", "Weights 1")

    print("\n🔄 Triggering .backward() on the Loss node...")
    
    # We want to find the order of execution required to resolve the Loss
    execution_path = autograd.topological_sort("Loss")
    
    print("\n✅ Valid Execution Order (Dependencies resolved first):")
    for step, node in enumerate(execution_path):
        print(f"Step {step + 1}: Compute -> {node}")