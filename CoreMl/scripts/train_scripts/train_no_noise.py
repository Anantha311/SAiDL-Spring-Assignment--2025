import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scripts.main.CNN_model import SimpleCNN

# Define transformations for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(), # Converts pixel value of [0,255] to [0,1]
    transforms.Normalize((0.5,), (0.5,)) # converts from [0,1]
])

data_path = "~/Projects/AI-ML/SAIDL/CoreMl/dataset"
# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "CE_no_noise.pth"
model = SimpleCNN().to(device)

# Create directory for model storage
noise_diretory = "no_noise"
number = str(1)
model_dir = os.path.join("/home/anantha/Projects/AI-ML/SAIDL/CoreMl/weights", number)
model_dir = os.path.join(model_dir, noise_diretory)
model_dir = os.path.join(model_dir, model_name)
os.makedirs(model_dir, exist_ok=True)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss() # Right now uses CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store loss and accuracy
losses = []
accuracies = []

# Train for 100 epochs
for epoch in range(20):  # Train for 100 epochs
    running_loss = 0.0 # Tracks total loss for the epoch
    correct = 0 # Counts correctly classified samples
    total = 0 # Tracks total number of samples
    model.train()  # Sets model to training mode (important for dropout & batch norm)
    
    for images, labels in train_loader: #loops through the mini-batches of training data.
        images, labels = images.to(device), labels.to(device) #Moves images and labels to GPU (cuda) or CPU (cpu) depending on device.

        optimizer.zero_grad() # Clears gradients from the previous batch
        outputs = model(images) #Passes the batch of images through the CNN to get predicted outputs (logits).
        loss = criterion(outputs, labels) # Computes the loss between predicted outputs and ground-truth labels.
        loss.backward() # Computes gradients of loss w.r.t. model parameters using automatic differentiation.
        optimizer.step() # Updates model parameters using the optimizer 

        running_loss += loss.item() # Adds the current batch loss to running_loss, accumulating total loss for the epoch.

        # Track accuracy
        _, predicted = torch.max(outputs, 1) # finds the class with the highest probability (prediction).
        correct += (predicted == labels).sum().item() #  counts the correct predictions in the batch.
        total += labels.size(0) #  Keeps track of the total images processed.

    epoch_loss = running_loss / len(train_loader) # Averages the loss over all batches in the epoch.
    epoch_acc = 100 * correct / total # Computes accuracy (%)
    losses.append(epoch_loss)
    accuracies.append(epoch_acc)
    print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")

# Save the trained model
model_path = os.path.join(model_dir, model_name) 
file_path =  os.path.join(model_path, model_name + ".pth")
torch.save(model.state_dict(),file_path )

# Plot Loss vs. Epochs
plt.figure()
plt.plot(range(1, 21), losses, label='Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.savefig(os.path.join(model_dir, "loss_vs_epochs.png"))

# Plot Accuracy vs. Epochs
plt.figure()
plt.plot(range(1, 21), accuracies, label='Accuracy', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Epochs')
plt.legend()
plt.savefig(os.path.join(model_dir, "accuracy_vs_epochs.png"))

print("Training complete! Model and graphs saved in", model_dir)
