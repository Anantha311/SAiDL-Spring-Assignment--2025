import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define transformations (already done in your code)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Load datasets
data_path = "~/Projects/AI-ML/SAIDL/CoreMl/dataset"
trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=transform) # This loads the dataset of train
testset = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=transform) # This loads the dataset of test

# Check dataset size
print(f"Training Set Size: {len(trainset)} images")
print(f"Test Set Size: {len(testset)} images")

# Show a few images
classes = trainset.classes  # Get class names 

def imshow(img):
    img = img / 2 + 0.5  # Unnormalize to [0,1] range basically earlier it was from [-1,1] and now it is [0,1]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # PyTorch tensors are in (C, H, W), matplotlib.pyplot.imshow() expects images in (H, W, C) format.
    plt.show() 

# Get a batch of training data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True) # Wraps trainset in a DataLoader to load batches of images.
data_iter = iter(trainloader) # Converts trainloader into an iterator, allowing us to fetch a batch.
images, labels = next(data_iter) # Retrieves the next batch (4 images and their labels).

# Display images
imshow(torchvision.utils.make_grid(images)) # arranges 4 images into a single grid.
print("Labels:", [classes[labels[i]] for i in range(4)])