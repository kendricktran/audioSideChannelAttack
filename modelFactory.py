'''
modelFactory.py

This module provides a factory class for creating and configuring different neural network models
for audio classification. It supports various architectures from torchvision.models including
ResNet, MobileNet, EfficientNet, DenseNet, and VGG variants.

The factory handles the modification of the final classification layer to match the number of
classes specified in the config file.

Usage:
    from modelFactory import ModelFactory
    
    # Create a ResNet18 model
    model = ModelFactory.get_model('resnet18')
    
    # Create a MobileNetV2 model
    model = ModelFactory.get_model('mobilenet_v2')

Supported Models:
    - resnet18
    - resnet50
    - mobilenet_v2
    - efficientnet_b0
    - densenet121
    - vgg11
    - vgg13
    - vgg16
    - vgg19

Note: All models are initialized with ImageNet pretrained weights and their final
classification layer is modified to match config.num_classes.
'''

import torch
import torch.nn as nn
from torchvision import models
from config import config

class ModelFactory:
    @staticmethod
    def get_model(model_name='resnet18'):
        """
        Factory method to get different model architectures.
        
        Args:
            model_name (str): Name of the model to use. Options:
                - 'resnet18'
                - 'resnet50'
                - 'mobilenet_v2'
                - 'efficientnet_b0'
                - 'densenet121'
                - 'vgg11'
                - 'vgg13'
                - 'vgg16'
                - 'vgg19'
        
        Returns:
            torch.nn.Module: The specified model with modified final layer
        """
        if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config.num_classes)
            
        elif model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config.num_classes)
            
        elif model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, config.num_classes)
            
        elif model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, config.num_classes)
            
        elif model_name == 'densenet121':
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, config.num_classes)
            
        elif model_name == 'vgg11':
            model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, config.num_classes)
            
        elif model_name == 'vgg13':
            model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, config.num_classes)
            
        elif model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, config.num_classes)
            
        elif model_name == 'vgg19':
            model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, config.num_classes)
            
        else:
            raise ValueError(f"Model {model_name} not supported. Please choose from: resnet18, resnet50, mobilenet_v2, efficientnet_b0, densenet121, vgg11, vgg13, vgg16, vgg19")
        
        return model 