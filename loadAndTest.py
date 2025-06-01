import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision import models
from keyLabel import keyLabel
from config import config
from coatnetNote import CoAtNet


'''
loadAndTest.py

This script loads the trained CoAtNet model and tests it on the testing data. It loads the mel spectrograms of the testing data and uses the model to predict the class of each mel spectrogram.
To run this you will have needed to save the weights of the model. This is done at the end of coatnetNote.py using torch.save(model.state_dict(), "model_weights.pth").

Usage:
    python3 loadAndTest.py

Arguments:
    None
'''

def main():
    # Load model weights
    model_weights = "resnet_model_weights_personal.pth"
    testing_folder = "test_data"

    total = 0
    total_correct = 0

    labeler = keyLabel()
    
    # Add this to ensure all class labels are known
    labeler.fit(['q','w','e','r','t','y','u','i','o','p'])  # Adjust this list to include all possible labels

    # Load the ResNet model
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.num_classes)
    # model = CoAtNet(config.img_size[0:2], 3, config.num_blocks, config.channels, num_classes=config.num_classes)
    model.load_state_dict(torch.load(model_weights))
    model.eval()

    os.system('python3 librosaPeaks.py -test')

    for file in os.listdir(testing_folder):
        if file.endswith(".npy") and file[0] != "*":
            x = np.load(os.path.join(testing_folder, file))
            x = cv2.resize(x, (224, 224))
            x = np.stack([x] * 3, axis=0)  # Convert to 3-channel
            x = torch.from_numpy(x)
            x = x.unsqueeze(0)
            logits = model(x.float())
            probabilities = F.softmax(logits, dim=1)
            #print(f"Probabilities: {probabilities}")  # Debugging line
            predicted_class = torch.argmax(probabilities, dim=1)
            predicted_class = labeler.inverse_transform([predicted_class.item()])[0]
            actual_class = file[0]
            if (predicted_class == actual_class):
                total_correct += 1

            total += 1
            print(f"Predicted Class: {predicted_class} vs. Actual Class: {actual_class}")
    print("Accuracy: ", total_correct / total if total > 0 else 0)

if __name__ == '__main__':
    main()