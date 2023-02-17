class Node:
    def __init__(self, feature: str, threshold: int, left: 'Node' = None, right: 'Node' = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right