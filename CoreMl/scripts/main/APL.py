
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class NCE_MAE:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def nce_loss(self, logits, targets, eps=1e-8):
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
        

    def mae_loss(self, logits, targets):
        probs = torch.softmax(logits, dim=1) # Makes all elements between 0 and 1 such that you sum each element in a row it becomes 1
        one_hot_targets = F.one_hot(targets, num_classes=logits.shape[1]).float()
        #torch.nn.functional.one_hot(tensor, num_classes=-1)
        #tensor: A tensor containing class indices (integers).
        #num_classes: The total number of classes (optional, but required for consistent sizing).
        return torch.abs(probs - one_hot_targets).mean()

    def __call__(self, logits, targets):
        active_loss = self.nce_loss(logits, targets)
        passive_loss = self.mae_loss(logits, targets)
        return (1 - self.alpha) * active_loss + self.alpha * passive_loss


class NFL_RCE:
    def __init__(self, alpha=0.5,gamma=2.0,smooth=0.1, clip_value=1e-7,reduction='mean'):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = smooth
        self.clip_value = clip_value

    def nfl_loss(self,logits,targets):
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
    
    def rce_loss(self, logits, targets):
        num_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)
        
        # Advanced error-proof smoothing
        with torch.no_grad():
            targets = targets.view(-1, 1)  #  Reshapes targets into a column vector of shape (batch_size, 1).
            smooth_targets = torch.full((targets.size(0), num_classes), self.smooth/(num_classes-1), device=logits.device)
            smooth_targets.scatter_(1, targets, 1-self.smooth)
            smooth_targets = torch.clamp(smooth_targets, self.clip_value, 1-self.clip_value)

        # Numerically stable computation
        loss = -torch.sum(probs * torch.log(smooth_targets), dim=1)
        return loss.mean()


    def __call__(self, logits, targets):
        active_loss = self.nfl_loss(logits, targets)
        passive_loss = self.rce_loss(logits, targets)
        return (1 - self.alpha) * active_loss + self.alpha * passive_loss



class NFL_MAE:
    def __init__(self, alpha=0.5,gamma=2.0,reduction='mean'):
        self.gamma=gamma
        self.alpha = alpha
        self.reduction = reduction

    def nfl_loss(self,logits,targets):
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
    
    def mae_loss(self, logits, targets):
        probs = torch.softmax(logits, dim=1) # Makes all elements between 0 and 1 such that you sum each element in a row it becomes 1
        one_hot_targets = F.one_hot(targets, num_classes=logits.shape[1]).float()
        #torch.nn.functional.one_hot(tensor, num_classes=-1)
        #tensor: A tensor containing class indices (integers).
        #num_classes: The total number of classes (optional, but required for consistent sizing).
        return torch.abs(probs - one_hot_targets).mean()

    def __call__(self, logits, targets):
        active_loss = self.nfl_loss(logits, targets)
        passive_loss = self.mae_loss(logits, targets)
        
        return (1 - self.alpha) * active_loss + self.alpha * passive_loss

class NCE_RCE:
    def __init__(self, alpha=0.5,smooth=0.1,clip_value=1e-7):
        self.smooth = smooth
        self.alpha = alpha
        self.clip_value = clip_value

    def nce_loss(self, logits, targets,eps=1e-8):
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
    
    def rce_loss(self, logits, targets):
        num_classes = logits.size(1)
        probs = torch.softmax(logits, dim=1)
        
        # Advanced error-proof smoothing
        with torch.no_grad():
            targets = targets.view(-1, 1)
            smooth_targets = torch.full((targets.size(0), num_classes), 
                                      self.smooth/(num_classes-1), 
                                      device=logits.device)
            smooth_targets.scatter_(1, targets, 1-self.smooth)
            smooth_targets = torch.clamp(smooth_targets, self.clip_value, 1-self.clip_value)

        # Numerically stable computation
        loss = -torch.sum(probs * torch.log(smooth_targets), dim=1)
        return loss.mean()

    def __call__(self, logits, targets):
        active_loss = self.nce_loss(logits, targets)
        passive_loss = self.rce_loss(logits, targets)
        
        return (1 - self.alpha) * active_loss + self.alpha * passive_loss
