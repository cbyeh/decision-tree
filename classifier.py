import numpy as np
import pandas as pd
from collections import deque

# A node in our decision tree
class node:
    def __init__(self, points):
        self.points = points # Points as a pandas dataframe
        self.rules = None # Rules for non-leaf nodes
        self.predict = None # Leaf node prediction
        self.left_child = None
        self.right_child = None
        self.is_done = False
        self.is_leaf = False
        # Get majority label
        # @property
        # def majority(self):
        #     return 1

# Create our decision tree
def ID3():
    # Initialize with all the training data at a single root node
    root = node(pd.read_csv('pa2train.txt', sep = r'\s+', header = None))
    # While there is an impure node in the tree:
    while root:
        # Pick any impure node
        curr = _pick_impure_node(root)
        if not curr: break
        # Get decision rule xt <= t TODO: get rid of data after decision is made, so node is pure
        index, split = _pick_decision_rule(curr.points)
        curr.rules = (index, split)
        left = node(curr.points[curr.points[index] <= split])
        right = node(curr.points[curr.points[index] > split])
        # If either child are pure, make them a leaf that predicts their single label
        if not _node_is_impure(left):
            left.predict = left.points[22].iloc[0]
            left.is_done = True
            left.is_leaf = True
        if not _node_is_impure(right):
            right.predict = right.points[22].iloc[0]
            right.is_done = True
            right.is_leaf = True
        curr.left_child = left
        curr.right_child = right
        curr.is_done = True
    return root

# See if a node has both labels
def _node_is_impure(node):
    if not node or node.is_done:
        return False
    labels = node.points.iloc[:, 22]
    return 1.0 in labels.values and 0.0 in labels.values

# Pick an impure node via a queue
def _pick_impure_node(root):
    de = deque()
    de.append(root)
    while(len(de) > 0):
        gone = de.pop()
        if _node_is_impure(gone):
            return gone
        if not gone.left_child.is_leaf:
            de.append(gone.left_child)
        if not gone.right_child.is_leaf:
            de.append(gone.right_child)
    return None

# Find the best decision rule given a node's data set via information gain and entropy calculation
def _pick_decision_rule(points):
    print("Deciding: ")
    best_index = -1
    best_gain = float('inf')
    best_split = None
    features_list = list(points) # List of 0-22
    features_list.pop() # Pop the labels column
    for index in features_list:
        # print(index)
        features = points.iloc[:, index] # Features in x_index
        # Get distinct values from features and sort
        distinct_features = sorted(features.unique())
        # Find best candidate from all possible splits of features at this index
        for i in range(len(distinct_features) - 1):
            split = (distinct_features[i] + distinct_features[i + 1]) / 2
            # Split the data to one that's less than or equal to split, and those greater than
            lte = features[(features <= split)]
            gt = features[(features > split)]
            # Calculate conditional entropy
            prob_lte = len(lte) / len(features)
            prob_gt = len(gt) / len(features)
            ig = prob_lte * _entropy(lte, points) + prob_gt * _entropy(gt, points)
            # print("Info gain: " + str(ig))
            if ig < best_gain:
                best_index = index
                best_gain = ig
                best_split = split
    print("Best index: " + str(best_index))
    print("Best gain: " + str(best_gain))
    print("Best split: " + str(best_split))
    return best_index, best_split

# Calculate the entropy of H(Y | Z = z)
def _entropy(split_features, points):
    zero_count = 0
    one_count = 0
    labels = points.iloc[:, 22]
    for index, _ in split_features.iteritems():
        if labels[index] == 0.0:
            zero_count += 1
        else:
            one_count += 1
    prob_zero = zero_count / len(split_features)
    prob_one = one_count / len(split_features)
    if prob_zero == 0:
        log_zero = 0.0
    else:
        log_zero = np.log2(prob_zero)
    if prob_one == 0:
        log_one = 0.0
    else:
        log_one = np.log2(prob_one)
    return -prob_zero * log_zero - prob_one * log_one

# Predict a label given the root of the tree, where point is the row with features
def predict(tree, point):
    curr = tree
    while not curr.is_leaf:
        index, split = curr.rules
        if (point[index] <= split):
            curr = curr.left_child
        else:
            curr = curr.right_child
    return curr.predict

# Find training, validation, or test error
def get_error(tree, filename):
    points = pd.read_csv(filename, sep = r'\s+', header = None)
    mistakes = 0
    total_points = 0
    for _, row in points.iterrows():
        if row[22] != predict(tree, row):
            mistakes += 1
        total_points += 1
    print("__ERROR_RATE__")
    print("Mistakes: " + str(mistakes))
    print("Total data points: " + str(total_points))
    return float(mistakes) / total_points

# Prune a tree with BFS.
def prune(tree):
    return 1

# Find error
if __name__ == "__main__":
    tree = ID3()
    print(get_error(tree, 'pa2test.txt'))

# TODO: Make more general for other datasets