import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor()])
data_path = "~/Projects/AI-ML/SAIDL/CoreMl/dataset"
trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

# Extract original data and labels
train_data = trainset.data  # Shape: (50000, 32, 32, 3)
train_labels = np.array(trainset.targets)  # Shape: (50000,)
test_data = testset.data  # Shape: (10000, 32, 32, 3)
test_labels = np.array(testset.targets)  # Shape: (10000,)
def introduce_symmetric_noise(labels, num_classes=10, noise_rate=0.2):
    """ Introduces symmetric label noise in a dataset. """
    np.random.seed(42) # This give the same random number whenever the seed is 42 
    noisy_labels = labels.copy()
    num_noisy = int(noise_rate * len(labels)) # get the number of noisy lables

    # Select indices to corrupt
    noisy_indices = np.random.choice(len(labels), num_noisy, replace=False) # Chooses num_noisy random number from [0,len(labels)]  
    for i in noisy_indices:
        original_label = labels[i] # What was the orginal label like aeroplane or etc
        possible_labels = list(range(num_classes)) # All available classes [0,9] both included
        possible_labels.remove(original_label)  # Remove correct label
        noisy_labels[i] = np.random.choice(possible_labels)  # Assign incorrect label

    return noisy_labels

# Introduce noise at eta = [0.2,0.8]
eta = 0.8
noisy_labels_train = introduce_symmetric_noise(train_labels, noise_rate=eta)
noisy_labels_test = introduce_symmetric_noise(test_labels, noise_rate=eta)

# Save dataset in PyTorch format
dataset_name = f"/home/anantha/Projects/AI-ML/SAIDL/CoreMl/dataset/cifar10_noisy_dataset_{eta}.pt"
torch.save({"images": torch.tensor(train_data, dtype=torch.float32),  # Ensure correct dtype
            "labels": torch.tensor(noisy_labels_train, dtype=torch.long),
            "test_images":torch.tensor(test_data, dtype=torch.float32),
            "test_labels" : torch.tensor(noisy_labels_test,dtype=torch.long)
            }, 
           dataset_name)
print(f"Dataset saved as {dataset_name}")

# Load dataset (PyTorch)
loaded_data = torch.load(dataset_name)
loaded_images, loaded_labels,loaded_test_images,loaded_test_labels = loaded_data["images"], loaded_data["labels"],loaded_data["test_images"], loaded_data["test_labels"]

print(f"Loaded images shape: {loaded_images.shape}")   # Expected: (50000, 32, 32, 3)
print(f"Loaded labels shape: {loaded_labels.shape}")   # Expected: (50000,)
print(f"Loaded images shape: {loaded_test_images.shape}")   # Expected: (10000, 32, 32, 3)
print(f"Loaded labels shape: {loaded_test_labels.shape}") # Expected: (10000,)