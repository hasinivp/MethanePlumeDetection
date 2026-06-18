import torch
import torch.nn as nn
import numpy as np
from unet import UNet
from dataset import MethaneDataset
import os
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

def get_centroid(mask):
    rows, cols = np.where(mask > 0)
    if len(rows) == 0 or len(cols) == 0:
        return None, None
    center_row = rows.mean()
    center_col = cols.mean()
    return center_row, center_col

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
ious = [] #list to store IoU values for each sample in the validation set

with torch.no_grad(): #no need to compute gradients during inference
    for image, mask in val_loader:
        output = model(image) #pass the images through the model
        predicted_mask = torch.sigmoid(output) #apply sigmoid to get probabilities
        pred_binary = (predicted_mask > 0.5).float() #threshold the probabilities to get binary masks
        center_row, center_col = get_centroid(pred_binary[0].squeeze(0).numpy()) #get the centroid of the predicted plume for visualization
        print("Centroid of predicted plume:", center_row, center_col) #print the centroid coordinates of the predicted plume
        intersection = (pred_binary * mask).sum(dim=(1, 2, 3)) #calculate intersection of predicted and ground truth masks
        union = (pred_binary + mask).clamp(0, 1).sum(dim=(1, 2, 3)) #calculate union of predicted and ground truth masks
        iou = intersection / union #calculate IoU metric
        print("IoU for current batch:", iou.mean().item()) #print the average IoU for the current batch
        ious.extend(iou.tolist())
        
        print("------------------")
'''
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(image[0].squeeze(0).numpy(), cmap="viridis")
        plt.title("Methane Image")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(mask[0].squeeze(0).numpy(), cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(pred_binary[0].squeeze(0).numpy(), cmap="gray")
        plt.title("Prediction")
        plt.axis("off")
        plt.scatter(center_col, center_row, color="red", marker="x", s=100) #mark the center of the predicted plume with a red "x"

        plt.show()
'''

print("Mean IoU:", np.mean(ious)) #print the mean IoU across all samples in the validation set
print("Min IoU:", np.min(ious)) #print the minimum IoU across all samples in the validation set
print("Max IoU:", np.max(ious)) #print the maximum IoU across all samples in the validation set



