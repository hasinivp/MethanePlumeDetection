import torch
import torch.nn as nn
import numpy as np
from unet import UNet
from dataset import MethaneDataset
import os
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

dataset = MethaneDataset("dataset/images_resized", "dataset/masks_resized")

dataloader = DataLoader(dataset, batch_size=4, shuffle=True) #batch size is how many samples to load at once, shuffle is to randomize the order of data

images, masks = next(iter(dataloader)) #get one batch of data
print("Images shape:", images.shape) #should be (batch_size, 1, height, width)
print("Masks shape:", masks.shape) #should be (batch_size, 1, height, width)

#train/validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size]) #split the dataset into training and validation sets
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

model = UNet()
model.load_state_dict(torch.load("unet_epoch_1.pth")) #load the model weights from the first epoch of training
model.eval() #set the model to evaluation mode

image, mask = val_dataset[0] #get one sample from the validation dataset
with torch.no_grad(): #no need to compute gradients during inference
    output = model(image.unsqueeze(0)) #add batch dimension and pass the image through the model
    predicted_mask = torch.sigmoid(output).squeeze(0).squeeze(0) #apply sigmoid to get probabilities, remove batch and channel dimensions
    pred_binary = (predicted_mask > 0.5).float() #threshold the probabilities to get binary mask

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(image.squeeze(0), cmap="viridis")
plt.title("Methane Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(mask.squeeze(0), cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(pred_binary, cmap="gray")
plt.title("Prediction")
plt.axis("off")

plt.show()