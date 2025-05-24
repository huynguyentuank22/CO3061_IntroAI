import numpy as np

class SumTree:
    """
    A sum tree data structure for efficient priority-based sampling.
    Used for Prioritized Experience Replay.
    """
    
    def __init__(self, capacity):
        """Initialize a sum tree with given capacity"""
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Tree nodes
        self.data_pointer = 0  # Pointer to next data index
        self.size = 0  # Current size
    
    def add(self, priority):
        """Add a new priority value to the tree"""
        # Get the leaf index
        tree_idx = self.data_pointer + self.capacity - 1
        
        # Update the tree with the new priority
        self.update(tree_idx, priority)
        
        # Update data pointer for next insertion
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        # Increment size until we reach capacity
        if self.size < self.capacity:
            self.size += 1
    
    def update(self, tree_idx, priority):
        """Update a priority value at a specific tree index"""
        # Compute the change in priority
        change = priority - self.tree[tree_idx]
        
        # Update the leaf node
        self.tree[tree_idx] = priority
        
        # Propagate the change up through the tree
        while tree_idx != 0:  # While not yet at the root
            # Move to parent
            tree_idx = (tree_idx - 1) // 2
            
            # Update the parent node
            self.tree[tree_idx] += change
    
    def get(self, value):
        """
        Get a leaf node and its corresponding data index given a value.
        
        Args:
            value: A value in range [0, total_priority].
            
        Returns:
            leaf_idx: The leaf node index.
            priority: The priority value stored in the leaf node.
        """
        parent_idx = 0  # Start from the root
        
        while True:
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            
            # If we reach a leaf node, we're done
            if left_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
                
            # Choose the direction (left or right) based on the value
            if value <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                value -= self.tree[left_idx]
                parent_idx = right_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return data_idx, self.tree[leaf_idx]
    
    def total(self):
        """Get the total priority (sum of all priorities)"""
        return self.tree[0] if self.size > 0 else 0
    
    def max(self):
        """Get the maximum priority in the tree"""
        return np.max(self.tree[-self.capacity:]) if self.size > 0 else 1.0
    
    def __len__(self):
        """Get the current size of the tree"""
        return self.size
    
    def clear(self):
        """Clear the tree"""
        self.tree.fill(0)
        self.data_pointer = 0
        self.size = 0
