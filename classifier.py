import numpy as np
import pandas as pd

# A node in our decision tree
class node:
    def __init__(self, points):
        self.points = points # Points as a pandas dataframe
        self.rules = None
        self.left_child = None
        self.right_child = None
        self.is_traversed = False
        # Get majority label
        @property
        def majority(self):
            return 1

# Create our decision tree
def ID3():
    # Initialize with all the training data at a single root node
    root = node(pd.read_csv('pa2train.txt', sep = r'\s+', header = None))
    # While there is an impure node in the tree:
    while root:
        # Pick any impure node
        curr = _pick_impure_node(root)
        print(curr.points)
        if not curr: break
        # Get decision rule xt <= t TODO: get rid of data after decision is made, so node is pure

    return root

# See if a node has both labels
def _node_is_impure(node):
    if not node or node.is_traversed:
        return False
    labels = node.points.iloc[:, 22]
    return 1.0 in labels.index and 0.0 in labels.index

# Pick an impure node via preorder traversal
def _pick_impure_node(root):
    if root and not root.is_traversed:
        if _node_is_impure(root):
            return root
        _pick_impure_node(root.left_child)
        _pick_impure_node(root.right_child)

ID3()

# Find the best decision rule given a node's data set
def _pick_decision_rule(points):
    feature_index = 0
    threshold = 0
    return feature_index, threshold

# Calculate the entropy
def _entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    return np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])

# Calculate information gain.
def _info_gain(data, split_attribute_name, target_name = "class"):
    # Calculate the entropy of the total dataset
    total_entropy = _entropy(data[target_name])
    # Calculate the values and the corresponding counts for the split attribute 
    vals, counts = np.unique(data[split_attribute_name], return_counts = True)
    # Calculate the weighted entropy
    weighted_entropy = np.sum([(counts[i]/np.sum(counts))*_entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    # Calculate the information gain
    return total_entropy - weighted_entropy

# Predict a label given the root of the tree
def predict(tree):
    return 1