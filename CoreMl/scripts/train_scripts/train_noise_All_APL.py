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
from scripts.main.APL import NCE_MAE
from scripts.main.APL import NFL_RCE
from scripts.main.APL import NFL_MAE
from scripts.main.APL import NCE_RCE


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
    noise = 0.8 + noise_iter *0.2
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
    loss_NCE_MAE = []
    loss_NCE_RCE = []
    loss_NFL_MAE = []
    loss_NFL_RCE = []
    accuracies_NCE_MAE = []
    accuracies_NCE_RCE = []
    accuracies_NFL_MAE = []
    accuracies_NFL_RCE = []
 # Joins weights with dataset
    for model_type in range(4):
        if model_type == 0:
            model_name = "APL_NCE_MAE_noisy_labels_" + str(noise)
            criterion = NCE_MAE(alpha=0.7)
        if model_type == 1:
            model_name = "APL_NCE_RCE_noisy_labels_" + str(noise)
            criterion = NCE_RCE(alpha=0.7)
        if model_type == 2:
            model_name = "APL_NFL_MAE_noisy_labels_" + str(noise)
            criterion = NFL_MAE(alpha=0.7)
        if model_type == 3:
            model_name = "APL_NFL_RCE_noisy_labels_" + str(noise)
            criterion = NFL_RCE(alpha=0.7)
        model = SimpleCNN().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,                # Initial learning rate
            momentum=0.9,          # Helps escape local minima
            weight_decay=5e-4,     # Strong L2 regularization
            nesterov=True          # Improves convergence
        )

        # With learning rate scheduling
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[30, 60],  # When to decay LR
            gamma=0.1             # LR reduction factor
        )
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
                #print("Test image shape before permute:", images.shape)
               
                optimizer.zero_grad()  
                outputs = model(images) 
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
        
        model_dir = "/home/anantha/Projects/AI-ML/SAIDL/CoreMl/weights"
        number = str(1)
         
        model_dir_with_number= os.path.join(model_dir, number)
        model_path = os.path.join(model_dir_with_number , name_dataset_)
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
            loss_NCE_MAE = losses.copy()
            accuracies_NCE_MAE = accuracies.copy()
        if model_type == 1:
            loss_NCE_RCE = losses.copy()
            accuracies_NCE_RCE = accuracies.copy()
        if model_type == 2:
            loss_NFL_MAE = losses.copy()
            accuracies_NFL_MAE = accuracies.copy()
        if model_type == 3:
            loss_NFL_RCE = losses.copy()
            accuracies_NFL_RCE = accuracies.copy()

    model_dir = "/home/anantha/Projects/AI-ML/SAIDL/CoreMl/weights/" + number
    graph_path = os.path.join(model_dir, name_dataset_)

    #Cant plot losses as they are not in same dimensions
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), loss_NCE_MAE, label="Loss NCE-MAE")
    plt.plot(range(1, epochs + 1), loss_NCE_RCE, label="Loss NCE-RCE")
    plt.plot(range(1, epochs + 1), loss_NFL_MAE, label="Loss NFL-MAE")
    plt.plot(range(1, epochs + 1), loss_NFL_RCE, label="Loss NFL-RCE")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Comparison")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(graph_path, "loss_vs_epochs.png"))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), accuracies_NCE_MAE, label="Accuracy NCE-MAE")
    plt.plot(range(1, epochs + 1), accuracies_NCE_RCE, label="Accuracy NCE-RCE")
    plt.plot(range(1, epochs + 1), accuracies_NFL_MAE, label="Accuracy NFL-MAE")
    plt.plot(range(1, epochs + 1), accuracies_NFL_RCE, label="Accuracy NFL-RCE")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Comparison for {noise} rate" )
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(graph_path, "accuracy_vs_epochs.png"))















