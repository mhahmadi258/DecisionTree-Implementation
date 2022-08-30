from unittest.main import main
import numpy as np
from decision_tree import DecisionTreeClassifire

if __name__ == "__main__":
    # y
    examples = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1])
    # x
    attributes = np.array([[1, 0, 0, 1, 1, 2, 0, 1, 0, 0],
                           [1, 0, 0, 1, 2, 0, 0, 0, 1, 2],
                           [0, 1, 0, 0, 1, 0, 0, 0, 2, 0],
                           [1, 0, 1, 1, 2, 0, 1, 0, 1, 1],
                           [1, 0, 1, 0, 2, 2, 0, 1, 0, 3],
                           [0, 1, 0, 1, 1, 1, 1, 1, 3, 0],
                           [0, 1, 0, 0, 0, 0, 1, 0, 2, 0],
                           [0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
                           [0, 1, 1, 0, 2, 0, 1, 0, 2, 3],
                           [1, 1, 1, 1, 2, 2, 0, 1, 3, 1],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [1, 1, 1, 1, 2, 0, 0, 0, 2, 2]])

    clf = DecisionTreeClassifire()
    clf.fit(attributes, examples, [0, 0, 0, 0, 1, 1, 0, 0, 2, 2])
    print(clf.predict(attributes[:4]))
    clf.show_tree()
