import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/anantha/Projects/AI-ML/SAIDL/CoreMl")
from scripts.main.CNN_model import SimpleCNN


class NCE(nn.Module):
    def forward(self, logits, targets, eps=1e-8):
#         logits (torch.Tensor): Raw model outputs (batch_size, num_classes)
            #These are the model's output scores for the true (positive) samples before applying softmax.
            #They represent how confident the model is about a particular class.

#         targets (torch.Tensor): True labels (batch_size,)

#         eps (float): Small value for numerical stability
     
        log_probs = torch.log_softmax(logits, dim=1) # Range becomes (-infinity,0] , Closer to 0 better the probability
        prob_true =  torch.gather(log_probs, 1, targets.unsqueeze(1)).squeeze(1)
        # torch.gather(input, dim, index)
        #input: The source tensor.
        #dim: The dimension along which to gather values.
        #index: A tensor containing the indices of elements to gather.
        #Torchdim=1 means we gather along columns (second axis).
        # Basically it gathers probabilty of the true label (batch_size) as unsqueeeze
        numerator = -torch.mean(prob_true)

        one_hot_targets = F.one_hot(targets, num_classes=logits.shape[1]).float()
        class_probs = one_hot_targets.mean(dim=0)
        torch.clamp(class_probs,min=eps)
        baseline_loss = -torch.sum(class_probs * torch.log(class_probs + eps))
        denominator =  baseline_loss       

        loss = numerator/(denominator + eps)
        return loss

class NFL(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(NFL, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self,logits,targets):
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get the predicted probability of the target class
        target_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        #self.eps  = torch.full((64,), 1e-8)
        #self.eps = self.eps.to(device=target_probs.device)

        eps = torch.tensor(1e-8, dtype=torch.float32, device=target_probs.device)
        

        # Clamp to avoid log(0)
        #target_probs = torch.clamp(target_probs, min=eps)
        
        # Compute the weight term (1 - p_t)^gamma
        weight = (1 - target_probs) ** self.gamma

        # Compute the unnormalized focal loss
        focal_loss = -weight * torch.log(target_probs + eps)  # Avoid log(0)
        
        # Normalize by the mean of all modulated probabilities
        normalization_factor = torch.mean(weight) + eps
        #print("Model output shape:", normalization_factor)
        #print("Target probs shape:", focal_loss.shape)
        focal_loss = focal_loss.div_(normalization_factor)

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # No reduction (returns per-sample loss)

# ------------------------------
class MAE(nn.Module):
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1) # Makes all elements between 0 and 1 such that you sum each element in a row it becomes 1
        one_hot_targets = F.one_hot(targets, num_classes=logits.shape[1]).float()
        #torch.nn.functional.one_hot(tensor, num_classes=-1)
        #tensor: A tensor containing class indices (integers).
        #num_classes: The total number of classes (optional, but required for consistent sizing).
        return torch.abs(probs - one_hot_targets).mean()

class RCE(nn.Module):
    def __init__(self, smooth=0.1, clip_value=1e-7):
        super().__init__()
        self.smooth = smooth
        self.clip_value = clip_value

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)
        
        # Advanced error-proof smoothing
        with torch.no_grad():
            targets = targets.view(-1, 1)
            smooth_targets = torch.full((targets.size(0), num_classes), self.smooth/(num_classes-1),  device=logits.device)
            # This creates a tensor of (targets.size(0), num_classes) and fills all of the items with self.smooth/(num_classes-1)
            smooth_targets.scatter_(1, targets, 1-self.smooth)
            # At dim = 1 with and targets as reference replaces the place at targets index them with 1-self.smooth which is 0.9
            smooth_targets = torch.clamp(smooth_targets, self.clip_value, 1-self.clip_value)
            # Clamps it so it stays in between (self.clip_value, 1-self.clip_value) which is approx (0,1)

        # Numerically stable computation
        loss = -torch.sum(probs * torch.log(smooth_targets), dim=1)
        return loss.mean()

class FL(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FL, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Get probability of the target class
        target_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Compute the modulating factor (1 - p_t)^gamma
        weight = (1 - target_probs) ** self.gamma
        
        # Compute focal loss
        focal_loss = -weight * torch.log(target_probs + 1e-8)  # Avoid log(0)
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # No reduction
