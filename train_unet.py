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



