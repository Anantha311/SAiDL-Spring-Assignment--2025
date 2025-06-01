import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, "/home/anantha/Projects/AI-ML/SAIDL/SAiDL-Spring-Assignment--2025/CoreMl")
# from scripts.main.CNN_model import SimpleCNN
# from scripts.main.APL import NCE_RCE
from scripts.main.New_model import CNN8Layer
from scripts.main.APL_New import NCE_MAE

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts pixel value of [0,255] to [0,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # Converts from [0,1] to [-1,1]
])

class NoisyCIFAR10(torch.utils.data.Dataset):
    def __init__(self, dataset, noisy_labels):
        self.dataset = dataset
        self.noisy_labels = noisy_labels  

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, _ = self.dataset[index]  
        label = self.noisy_labels[index]  
        return image, label
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise = 0.6
data_path = "/home/anantha/Projects/AI-ML/SAIDL/SAiDL-Spring-Assignment--2025/CoreMl/dataset"
trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
name_dataset = "cifar10_noisy_dataset_" + str(noise) +".pt"
name_dataset_ = "cifar10_noisy_dataset_" + str(noise)
noisy_data_dir = os.path.join(data_path ,name_dataset )
noisy_data = torch.load( noisy_data_dir )
noisy_labels = noisy_data["labels"]  
noisy_trainset = NoisyCIFAR10(trainset, noisy_labels)
train_loader = torch.utils.data.DataLoader(noisy_trainset, batch_size=128, shuffle=True)
model_dir = "/home/anantha/Projects/AI-ML/SAIDL/SAiDL-Spring-Assignment--2025/CoreMl/weights"
number = str(1)

model_name = "NCE_MAE_noisy_labels_0.6"
criterion = NCE_MAE()

model_dir_with_number = os.path.join(model_dir, number )
model_path = os.path.join(model_dir_with_number, name_dataset_)
model_path = os.path.join(model_path,model_name) # joins the weights/dataset to framework_loss
model = CNN8Layer().to(device)
os.makedirs(model_path, exist_ok=True)
epochs =60
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=epochs
)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

losses = []
accuracies = []



for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()  

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  
        outputs = model(images)  
        # print("Model output logits:", outputs.min().item(), outputs.max().item(), outputs.mean().item())
        # print("Any NaN in logits?", torch.isnan(outputs).any().item())

        loss = criterion(outputs, labels)  # Fix: Pass noise_logits to loss function
        loss.backward()  
        optimizer.step()  
                        
   

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    scheduler.step() 
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    losses.append(epoch_loss)
    accuracies.append(epoch_acc)
    print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")

model_dir = "/home/anantha/Projects/AI-ML/SAIDL/SAiDL-Spring-Assignment--2025/CoreMl/weights"

model_dir_with_number = os.path.join(model_dir, number )
model_path = os.path.join(model_dir_with_number, name_dataset_)
model_path = os.path.join(model_path,model_name) # joins the weights/dataset to framework_loss
os.makedirs(model_path ,exist_ok=True) 
file_location = os.path.join(model_path, model_name + ".pth")
torch.save(model.state_dict(), file_location )

with open(os.path.join(model_path, "metrics.txt"), "w") as f:
    f.write(f"Final Test Accuracy: {accuracies[-1]:.2f}%\n")
    f.write(f"Final Test Loss: {losses[-1]:.4f}\n")

plt.figure()
plt.plot(range(1, epochs +1), losses, label='Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.savefig(os.path.join(model_path, "loss_vs_epochs.png"))


plt.figure()
plt.plot(range(1, epochs + 1), accuracies, label='Accuracy', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Epochs')
plt.legend()
plt.savefig(os.path.join(model_path, "accuracy_vs_epochs.png"))

print(f"✅ Training complete! Model and results saved in {model_path}")
