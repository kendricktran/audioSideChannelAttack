import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from melDataset import melDataset
from keyLabel import keyLabel
from config import config
import numpy as np
import os
from sys import argv
import cv2

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Modify the final layer to match the number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, config.num_classes)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optim = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

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

def train_epoch(model, data_loader):
    model.train()
    train_loss, train_correct = 0.0, 0

    for step, batch in enumerate(data_loader):
        optim.zero_grad()
        x, y = batch
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)  # Convert to 3-channel
        logits = model(x.float())
        loss = loss_fn(logits, y)
        train_loss += loss.item()
        loss.backward()
        optim.step()

        preds = torch.argmax(logits, dim=1).flatten()
        correct_preds_n = (preds == y).cpu().sum().item()
        train_correct += correct_preds_n
    return train_loss, train_correct

def valid_epoch(model, data_loader):
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
    if (len(argv) > 1):
        training_data_directory = argv[1]
    else:
        training_data_directory = 'mel_spectrograms_(128x321)'

    fold_history = {}

    training_data = load_training_data(training_data_directory)

    dataset = melDataset(training_data)

    splits = KFold(n_splits=config.num_splits, shuffle=True, random_state=1337)

    for fold, (train_idx, val_idx) in enumerate(splits.split(dataset)):
        print('Fold {}'.format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=test_sampler)

        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

        for epoch in range(config.epochs):
            torch.cuda.empty_cache()
            print('---train:')
            train_loss, train_correct = train_epoch(model, train_loader)
            print('\n---eval:')
            test_loss, test_correct = valid_epoch(model, test_loader)
            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100
            print('\n---status:')
            print("\tEpoch:{}/{} \n\tAverage Training Loss:{:.4f}, Average Test Loss:{:.4f}; \n\tAverage Training Acc {:.2f}%, Average Test Acc {:.2f}%\n".format(epoch + 1,
                                                                                                                                                                    config.epochs,
                                                                                                                                                                    train_loss,
                                                                                                                                                                    test_loss,
                                                                                                                                                                    train_acc,
                                                                                                                                                                    test_acc))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

        fold_history[f'fold{fold+1}'] = history
    torch.save(model.state_dict(), "resnet_model_weights.pth")
    np.save('resnet_history.npy', fold_history)

if __name__ == "__main__":
    main()
