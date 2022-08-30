import numpy as np


class Node:
    def __init__(self):
        self.children = list()
        self.attribute = None
        self.label = None


class DecisionTreeClassifire:
    """
    A binary decision tree classifire with discrete input and output.
    """

    def __init__(self):
        self.root = None

    def plurality(self, examples):
        """
        Determine which class is in the majority.
        """
        p = np.sum(examples == 1)
        n = np.sum(examples == 0)
        plurality = 1 if p >= n else 0
        return plurality

    def entropy(self, p):
        """
        Calculate binary entropy.
        """
        if p == 0 or p == 1:
            return 0.0
        return -(p*np.log2(p) + (1-p)*np.log2(1-p))

    def reminder(self, examples, attributes, attribute_idx):
        """
        Calculate reminder.
        """
        values = np.unique(attributes[:, attribute_idx])
        reminder = 0
        for value in values:
            search_mask = attributes[:, attribute_idx] == value
            pk = np.sum(examples[search_mask] == 1)
            nk = np.sum(examples[search_mask] == 0)
            reminder += ((pk+nk)/examples.size)*self.entropy(pk/(pk + nk))
        return reminder

    def gain(self, examples, attributes, attribute_idx):
        """
        Calculate gain by using entropy and reminder.
        This function will be used for finding best feature in decision tree algorithm.
        """
        p = np.sum(examples == 1)
        n = np.sum(examples == 0)
        gain = self.entropy(p/(p+n)) - \
            self.reminder(examples, attributes, attribute_idx)
        return gain

    def decision_tree(self, examples, attributes, attribute_map, parent_examples):
        """
        decision tree algorithm. 
        """
        node = Node()

        # Child node is empty
        if examples.size == 0:
            label = self.plurality(parent_examples)
            node.label = label
            return node

        # Child node only contains samples from single class
        elif np.sum(examples == 0) == 0 or np.sum(examples == 1) == 0:
            label = self.plurality(examples)
            node.label = label
            return node

        # All features are used
        elif attributes.size == 0:
            label = self.plurality(examples)
            node.label = label
            return node

        else:
            gains = list()
            for i in range(attributes.shape[1]):
                gains.append(self.gain(examples, attributes, i))
            attribute_idx = np.argmax(gains)
            valid_idx, values = attribute_map.pop(attribute_idx)
            node.attribute = valid_idx

            att = np.delete(attributes, attribute_idx, 1)

            for value in values:
                search_mask = attributes[:, attribute_idx] == value
                new_examples = examples[search_mask]
                new_attributes = att[search_mask]
                child_node = self.decision_tree(
                    new_examples, new_attributes, list(attribute_map), examples)
                node.children.append((value, child_node))
            return node

    def fit(self, X, Y, feature_values):
        """
        Build a decision tree from dataset (X, Y)

        parameters:
        -----------
        feature_values: A list with length of M, which is the number of features, determine the maximum value+1 for each feature.
        This means that the value could be in the range of [0, maximum value+1).
        For example, [3, 2] means that samples of dataset have 2 feature, The first one has 3+1 values 0,1,2,3 and the
        one has 2+1 values 0, 1 ,and 2.
        """
        attribute_map = list()
        for i in range(X.shape[1]):
            attribute_map.append(
                (i, [j for j in range(feature_values[i] + 2)]))
        node = self.decision_tree(Y, X, attribute_map, None)
        self.root = node

    def predict_sample(self, x, node):
        """
        Predict the label of only one sample.
        """
        if node.label is not None:
            return node.label

        child_node = None
        att = node.attribute
        for child in node.children:
            if x[att] == child[0]:
                child_node = child[1]
                break
        if child_node is None:
            raise Exception(f'invalid value {x[att]} for attribute {att}')
        label = self.predict_sample(x, child_node)
        return label

    def predict(self, x):
        """
        Predict labels of multiple samples.
        """
        root = self.root
        if root is None:
            raise Exception('The model first should be trained !!!!')
        labels = np.zeros(x.shape[0])
        for i, sample in enumerate(x):
            labels[i] = self.predict_sample(sample, root)
        return labels

    def show_tree_rec(self, node, value, ind):
        """
        This method is used in show_tree method.
        """
        if node.label is not None:
            print('\t'*ind, f'{value} ----> label: {node.label}')
        else:
            print('\t'*ind, f'{value} ----> att: {node.attribute}')
            for val, child in node.children:
                self.show_tree_rec(child, val, ind+1)

    def show_tree(self):
        """
        Display the tree in terminal.
        """
        self.show_tree_rec(self.root, '', 0)
