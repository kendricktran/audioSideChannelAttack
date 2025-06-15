'''
transfer_learn.py

This script implements transfer learning for audio classification using various pretrained models.
It trains a model to classify keystrokes from mel spectrograms of audio recordings.

The script performs the following steps:
1. Loads and preprocesses mel spectrograms from a training directory
2. Initializes a pretrained model using ModelFactory
3. Splits data into training and validation sets
4. Trains the model using transfer learning
5. Saves the trained model weights and training history

Usage:
    python3 transfer_learn.py [training_data_directory] [model_name]
    
    Arguments:
        training_data_directory (optional): Path to directory containing training data
                                          Default: 'training_data_2'
        model_name (optional): Name of the model architecture to use
                             Default: 'resnet18'
                             Options: resnet18, resnet50, mobilenet_v2, efficientnet_b0,
                                     densenet121, vgg11, vgg13, vgg16, vgg19

Requirements:
    - Training data organized in subdirectories by class
    - Each class directory should contain .npy files of mel spectrograms
    - modelFactory.py for model initialization
    - melDataset.py for data loading
    - keyLabel.py for label encoding
    - config.py for training parameters

Output:
    - {model_name}_model_weights.pth: Trained model weights
    - {model_name}_history.npy: Training history (loss and accuracy metrics)
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from melDataset import melDataset
from keyLabel import keyLabel
from config import config
from modelFactory import ModelFactory
import numpy as np
import os
from sys import argv
import cv2

# Initialize label encoder
label_encoder = keyLabel()

def load_training_data(training_data_directory):
    mel_spectrograms = []
    labels = []
    for note_folder in os.listdir(training_data_directory):
        if os.path.isdir(os.path.join(training_data_directory, note_folder)):
            note_path = os.path.join(training_data_directory, note_folder)
            for file_name in os.listdir(note_path):
                if file_name.endswith('.npy'):  # assuming mel spectrograms are in .npy format
                    file_path = os.path.join(note_path, file_name)
                    mel_spectrogram = np.load(file_path)  # load mel spectrogram using NumPy
                    mel_spectrograms.append(mel_spectrogram)
                    label = str(note_folder)[0]
                    labels.append(label)
    
    label_encoder.fit(labels)
    labels = label_encoder.transform(labels)
    
    return list(zip(mel_spectrograms, labels))

def train_epoch(model, data_loader, loss_fn, optimizer):
    model.train()
    train_loss, train_correct = 0.0, 0

    for step, batch in enumerate(data_loader):
        optimizer.zero_grad()
        x, y = batch
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)  # Convert to 3-channel
        logits = model(x.float())
        loss = loss_fn(logits, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1).flatten()
        correct_preds_n = (preds == y).cpu().sum().item()
        train_correct += correct_preds_n
    return train_loss, train_correct

def valid_epoch(model, data_loader, loss_fn):
    model.eval()
    val_loss, val_correct = 0.0, 0

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            x, y = batch
            x = x.unsqueeze(1).repeat(1, 3, 1, 1)  # Convert to 3-channel
            logits = model(x.float())
            loss = loss_fn(logits, y)
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1).flatten()
            correct_preds_n = (preds == y).cpu().sum().item()
            val_correct += correct_preds_n

    return val_loss, val_correct

def main():
    if (len(argv) > 2):
        training_data_directory = argv[1]
        model_name = argv[2]
    else:
        training_data_directory = 'training_data_2'
        model_name = 'resnet18'  # default model

    # Initialize model, loss function, and optimizer
    model = ModelFactory.get_model(model_name)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Load and prepare data
    training_data = load_training_data(training_data_directory)
    dataset = melDataset(training_data)

    # Split dataset into train and validation sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(config.epochs):
        torch.cuda.empty_cache()
        print('---train:')
        train_loss, train_correct = train_epoch(model, train_loader, loss_fn, optimizer)
        print('\n---eval:')
        val_loss, val_correct = valid_epoch(model, val_loader, loss_fn)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset) * 100
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset) * 100
        
        print('\n---status:')
        print(f"\tEpoch:{epoch + 1}/{config.epochs}")
        print(f"\tAverage Training Loss:{train_loss:.4f}, Average Validation Loss:{val_loss:.4f}")
        print(f"\tAverage Training Acc {train_acc:.2f}%, Average Validation Acc {val_acc:.2f}%\n")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    # Save model and history with model name in filename
    torch.save(model.state_dict(), f"{model_name}_model_weights.pth")
    np.save(f'{model_name}_history.npy', history)

if __name__ == "__main__":
    main()
