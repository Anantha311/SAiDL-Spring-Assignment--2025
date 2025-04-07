import torchvision
import torchvision.transforms as transforms

# Define dataset transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Set the path where CIFAR-10 will be downloaded
data_path = "~/Projects/AI-ML/SAIDL/CoreMl/dataset"

# Download and load CIFAR-10 dataset into the specified directory
trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)





