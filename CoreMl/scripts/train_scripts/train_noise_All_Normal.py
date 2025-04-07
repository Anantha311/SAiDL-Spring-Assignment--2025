import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, "/home/anantha/Projects/AI-ML/SAIDL/CoreMl")
from scripts.main.CNN_model import SimpleCNN
from scripts.main.loss_fun import NCE
from scripts.main.loss_fun import NFL
from scripts.main.loss_fun import FL




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

for noise_iter in range(1):
    noise = 0.6 + noise_iter*0.2
    noise = round(noise, 10)
    data_path = "/home/anantha/Projects/AI-ML/SAIDL/CoreMl/dataset"
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    name_dataset = "cifar10_noisy_dataset_" + str(noise) +".pt"
    name_dataset_ = "cifar10_noisy_dataset_" + str(noise)
    noisy_data_dir = os.path.join(data_path ,name_dataset )
    noisy_data = torch.load( noisy_data_dir ) # Load noisy labels
    noisy_labels = noisy_data["labels"]  
    noisy_trainset = NoisyCIFAR10(trainset, noisy_labels)
    train_loader = torch.utils.data.DataLoader(noisy_trainset, batch_size=64, shuffle=True)
    loss_NCE = []
    loss_NFL = []
    loss_CE = []
    loss_FL = []
    accuracies_NCE = []
    accuracies_NFL = []
    accuracies_CE = []
    accuracies_FL = []
 # Joins weights with dataset
    for model_type in range(4):
        if model_type == 0:
            model_name = "NCE_noisy_labels_" + str(noise)
            criterion = NCE()
        if model_type == 1:
            model_name = "NFL_noisy_labels_" + str(noise)
            criterion = NFL()
        if model_type == 2:
            model_name = "CE_noisy_labels_" + str(noise)
            criterion = nn.CrossEntropyLoss()
        if model_type == 3:
            model_name = "FL_noisy_labels_" + str(noise)
            criterion = FL()
        model = SimpleCNN().to(device)
        # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        losses = []
        accuracies = []
        epochs = 30
        print(f"Starting training of {model_name}")
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            model.train()  

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device) 
                optimizer.zero_grad()  
                outputs = model(images) 
                loss = criterion(outputs, labels) 
                loss.backward()  
                optimizer.step()  
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)  
                correct += (predicted == labels).sum().item()
          
                total += labels.size(0)
            # scheduler.step() 
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            losses.append(epoch_loss)
            accuracies.append(epoch_acc)
            print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")
        
        number = str(1)
        model_dir = "/home/anantha/Projects/AI-ML/SAIDL/CoreMl/weights/" + number
        model_path = os.path.join(model_dir, name_dataset_)
        model_path = os.path.join(model_path,model_name) # joins the weights/dataset to framework_loss
        os.makedirs(model_path ,exist_ok=True) 
        file_location = os.path.join(model_path, model_name + ".pth")
        torch.save(model.state_dict(), file_location) # The second parameter is file location so directory must exist u
        with open(os.path.join(model_path, "metrics.txt"), "w") as f:
            f.write(f"Final Train Accuracy: {accuracies[-1]:.2f}%\n")
            f.write(f"Final Train Loss: {losses[-1]:.4f}\n")
        plt.figure()
        plt.plot(range(1, epochs +1), losses, label='Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss vs. Epochs for {model_name}')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(model_path, "loss_vs_epochs.png"))


        plt.figure()
        plt.plot(range(1, epochs + 1), accuracies, label='Accuracy', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy vs. Epochs for {model_name}')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(model_path, "accuracy_vs_epochs.png"))

        print(f"✅ Training complete! Model and results saved in {model_path}")
        if model_type == 0:
            loss_NCE = losses.copy()
            accuracies_NCE = accuracies.copy()
        if model_type == 1:
            loss_NFL = losses.copy()
            accuracies_NFL = accuracies.copy()
        if model_type == 2:
            loss_CE = losses.copy()
            accuracies_CE = accuracies.copy()
        if model_type == 3:
            loss_FL = losses.copy()
            accuracies_FL = accuracies.copy()

    model_dir = "/home/anantha/Projects/AI-ML/SAIDL/CoreMl/weights/" + number
    graph_path = os.path.join(model_dir, name_dataset_)

    #Cant plot losses as they are not in same dimensions
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_NCE, label="Loss NCE")
    plt.plot(range(1, epochs + 1), loss_NFL, label="Loss NFL")
    plt.plot(range(1, epochs + 1), loss_CE, label="Loss CE")
    plt.plot(range(1, epochs + 1), loss_FL, label="Loss FL")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Comparison")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(graph_path, "loss_vs_epochs.png"))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), accuracies_NCE, label="Accuracy NCE")
    plt.plot(range(1, epochs + 1), accuracies_NFL, label="Accuracy NFL")
    plt.plot(range(1, epochs + 1), accuracies_CE, label="Accuracy CE")
    plt.plot(range(1, epochs + 1), accuracies_FL, label="Accuracy FL")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Comparison for {noise} rate" )
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(graph_path, "accuracy_vs_epochs.png"))










