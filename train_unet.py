import torch
import torch.nn as nn
import numpy as np
from dataset import MethaneDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

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

print("Training set size:", len(train_dataset))
print("Validation set size:", len(val_dataset))
'''

from unet import Down
import torch
down = Down(64, 128) #input channels is 1 for grayscale images, output channels is 64 for the first layer of the UNet
x = torch.randn(1, 64, 256, 256) #batch size of 1, 64 channels, height and width of 256
y = down(x)
print("Output shape:", y.shape) #should be (1, 128, 128, 128) because of the max pooling and convolution operations

from unet import Up
up = Up(1024, 512) #input channels is 1024 for the second layer of the UNet, output channels is 512 for the first layer of the UNet
x1 = torch.randn(1, 1024, 16, 16) #batch size of 1, 1024 channels, height and width of 16
x2 = torch.randn(1, 512, 32, 32) #batch size of 1, 512 channels, height and width of 32
y = up(x1, x2)
print("Output shape:", y.shape) #should be (1, 512, 32, 32) because of the upsampling and convolution operations
'''
from unet import UNet
model = UNet()
x=torch.randn(1, 1, 256, 256) #batch size of 1, 1 channel for grayscale images, height and width of 256
y = model(x)
print("Output shape:", y.shape) #should be (1, 1, 256, 256) because of the convolution operations in the UNet architecture

criterion = nn.BCEWithLogitsLoss() #binary cross entropy loss with logits, used for binary segmentation tasks - how wrong is the model 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) #Adam optimizer, learning rate of 0.0001

for epoch in range(1): #train for 1 epoch
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        optimizer.zero_grad() #zero the gradients before backpropagation
       
        outputs = model(images) #forward pass through the model
        loss = criterion(outputs, masks) #compute the loss
       
        loss.backward() #backpropagation to compute the gradients
        optimizer.step() #update the weights
        running_loss += loss.item() 
    avg_loss = running_loss / len(train_loader) #average loss for the epoch
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad(): #no need to compute gradients during validation
        for images, masks in val_loader:
            outputs = model(images) #forward pass through the model
            loss = criterion(outputs, masks) #compute the loss
            val_loss += loss.item()
    val_avg_loss = val_loss / len(val_loader) #average validation loss for the epoch
    print(f"Epoch {epoch+1}, Training Loss: {avg_loss:.4f}")
    print(f"Epoch {epoch+1}, Validation Loss: {val_avg_loss:.4f}")
    torch.save(model.state_dict(), f"unet_epoch_{epoch+1}.pth") #save the model weights after each epoch

'''
images, masks = next(iter(train_loader)) #get one batch of training data
    optimizer.zero_grad() #zero the gradients before backpropagation
   
    outputs = model(images) #forward pass through the model
    loss = criterion(outputs, masks) #compute the loss
   
    loss.backward() #backpropagation to compute the gradients
    optimizer.step() #update the weights
    print("Loss:", loss.item()) #print the loss value
'''