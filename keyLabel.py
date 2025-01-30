'''
keyLabel.py

This file contains the keyLabel class, which is used to transform labels to integers and vice versa.
'''

class keyLabel:
    def __init__(self):
        self.label_to_int = {}
        self.int_to_label = {}

    def fit(self, labels):
        unique_labels = sorted(set(labels))
        self.label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        self.int_to_label = {idx: label for label, idx in self.label_to_int.items()}

    def transform(self, labels):
        return [self.label_to_int[label] for label in labels]

    def inverse_transform(self, ints):
        return [self.int_to_label[i] for i in ints]