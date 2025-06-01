'''
config.py

This file contains the configuration parameters for the CoAtNet model.
'''

class config:
    img_size = (224, 224, 1) # Each image is 224x224 pixels with 1 channel
    num_splits = 2 # Represents the number of splits for KFold
    epochs = 8
    num_classes = 10 # Represents the number of classes to pick from (a-z)
    batch_size = 8
    weight_decay = 1e-8 # taken from the paper
    lr = 5e-5 # taken from the paper
    num_blocks = [2, 2, 3, 5, 2]            # L
    channels = [64, 96, 192, 384, 768]      # D