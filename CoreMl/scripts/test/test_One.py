import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import os
import sys
sys.path.insert(0, "/home/anantha/Projects/AI-ML/SAIDL/CoreMl")
from scripts.main.CNN_model import SimpleCNN
from scripts.main.CNN_model import SimpleCNN
from scripts.main.APL import NCE_MAE
from scripts.main.APL import NFL_RCE
from scripts.main.APL import NFL_MAE
from scripts.main.APL import NCE_RCE
from scripts.main.loss_fun import NCE
from scripts.main.loss_fun import NFL
from scripts.main.loss_fun import FL
import numpy as np


# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts pixel value of [0,255] to [0,1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # Converts from [0,1] to [-1,1]
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise = 0.8
noise = round(noise, 10)

data_path = "/home/anantha/Projects/AI-ML/SAIDL/CoreMl/dataset"
testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

name_dataset = "cifar10_noisy_dataset_" + str(noise) +".pt"
name_dataset_ = "cifar10_noisy_dataset_" + str(noise) 
noisy_data_dir = os.path.join(data_path ,name_dataset )

model_name = 'NCE_noisy_lables_' +str(noise)
criterion = NCE_RCE(alpha=0.5)

number = str(1)
model_dir = "/home/anantha/Projects/AI-ML/SAIDL/CoreMl/weights/" + number 
model_name_2 = model_name + '.pth'
model_path = os.path.join(model_dir,name_dataset_) # With mode_dir/name_dataset
model_path_1 = os.path.join(model_path,model_name)# With mode_dir/name_dataset/loss_model
model_path_2 = os.path.join(model_path_1,model_name_2) # With mode_dir/name_dataset/loss_model/loss_model.pth
model = SimpleCNN()
#print(f"THis is the model:{model.state_dict()}")
model.to(device)
model.load_state_dict(torch.load(model_path_2, map_location=device))
model.eval()
# Initialize Metrics
correct = 0
total = 0

total_loss = 0.0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute Performance Metrics
test_accuracy = 100 * correct / total
test_loss = total_loss / len(test_loader)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
results_path = os.path.join(model_path_1 , "test_results.txt")
with open(results_path, "w") as f:
    f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")

print(f"Results saved at: {results_path}")
       
