
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
from scripts.main.CNN_model import SimpleCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class NCE_MAE:
    def __init__(self, alpha=0.6,beta = 0.4):
        self.alpha = alpha
        self.beta = beta

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
        numerator = -prob_true
        denominator = -torch.sum(log_probs, dim=1)       

        loss = numerator/denominator
        return loss.mean()



    def mae_loss(self, logits, targets):
        probs = torch.softmax(logits, dim=1) # Makes all elements between 0 and 1 such that you sum each element in a row it becomes 1
        one_hot_targets = F.one_hot(targets, num_classes=logits.shape[1]).float()
        #torch.nn.functional.one_hot(tensor, num_classes=-1)
        #tensor: A tensor containing class indices (integers).
        #num_classes: The total number of classes (optional, but required for consistent sizing).
        return torch.abs(probs - one_hot_targets).mean() #PyTorch’s .mean() without any arguments computes the mean of all elements in the tensor — across all dimensions.

    def __call__(self, logits, targets):
        active_loss = self.nce_loss(logits, targets)
        passive_loss = self.mae_loss(logits, targets)
        loss = self.alpha * active_loss + self.beta * passive_loss
        # print(f'Active loss = {active_loss}, Passive loss = {passive_loss}, Loss = {loss}')
        return loss


class NFL_RCE:
    def __init__(self, alpha=0.5,beta=0.5,gamma=2.0,smooth=0.1, clip_value=1e-7,reduction='mean'):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = smooth
        self.clip_value = clip_value
        self.beta = beta
        self.small_value=1e-4
    def nfl_loss(self,logits,targets):
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        target_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        target_probs = torch.clamp(target_probs, min=self.small_value, max=1.0)

        # Get the predicted probability of the target class
        #self.eps  = torch.full((64,), 1e-8)
        #self.eps = self.eps.to(device=target_probs.device)
        eps = torch.tensor(1e-8, dtype=torch.float32, device=target_probs.device)
        log_probs = torch.log(torch.clamp(probs, min=self.small_value))

        num_weight = (1 - target_probs) ** self.gamma
        numerator = -num_weight * torch.log(target_probs + eps)
        # Clamp to avoid log(0)
        #target_probs = torch.clamp(target_probs, min=eps)
        
        # Compute the weight term (1 - p_t)^gamma
        

        # Compute the unnormalized focal loss
        denom_weights = (1 - probs) ** self.gamma 
        a = denom_weights * log_probs
        denominator = torch.clamp(-torch.sum(a, dim=1), min=self.small_value)
        focal_loss = (numerator / denominator).mean()
        # print("probs min/max/mean:", probs.min().item(), probs.max().item(), probs.mean().item())
        # print("target_probs min/max:", target_probs.min().item(), target_probs.max().item())
        # print("numerator any nan?", torch.isnan(numerator).any().item())
        # print("denominator any nan?", torch.isnan(denominator).any().item())
        return focal_loss

        
    def rce_loss(self, logits, targets):
        probs = torch.softmax(logits, dim=1)  # p(k|x)
        batch_size, num_classes = probs.shape

        with torch.no_grad():
            q = torch.full_like(probs, fill_value=self.small_value)  # q(k|x) ≈ 1e-4
            q.scatter_(1, targets.view(-1, 1), 1.0)  # q(y|x) = 1
            q = torch.clamp(q, min=self.small_value, max=1.0)
        
        loss = -torch.sum(probs * torch.log(q), dim=1)
        return loss.mean()

    def __call__(self, logits, targets):
        # print("Logits stats:", logits.min().item(), logits.max().item(), logits.mean().item())
        # print("Targets stats:", targets.min().item(), targets.max().item())

        active_loss = self.nfl_loss(logits, targets)
        passive_loss = self.rce_loss(logits, targets)

        loss = self.alpha * active_loss + self.beta * passive_loss
        # print(f'Active loss: {active_loss}, Passive loss: {passive_loss}, Loss: {loss}')
        return loss




class NFL_MAE:
    def __init__(self, alpha=0.5,beta=0.5,gamma=2.0,reduction='mean'):
        self.gamma=gamma
        self.alpha = alpha
        self.reduction = reduction
        self.beta = beta
        self.small_value=1e-4

    def nfl_loss(self,logits,targets):
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        target_probs = probs.gather(1, targets.view(-1, 1)).squeeze(1)
        target_probs = torch.clamp(target_probs, min=self.small_value, max=1.0)

        # Get the predicted probability of the target class
        #self.eps  = torch.full((64,), 1e-8)
        #self.eps = self.eps.to(device=target_probs.device)
        eps = torch.tensor(1e-8, dtype=torch.float32, device=target_probs.device)
        log_probs = torch.log(torch.clamp(probs, min=self.small_value))

        num_weight = (1 - target_probs) ** self.gamma
        numerator = -num_weight * torch.log(target_probs + eps)
        # Clamp to avoid log(0)
        #target_probs = torch.clamp(target_probs, min=eps)
        
        # Compute the weight term (1 - p_t)^gamma
        

        # Compute the unnormalized focal loss
        denom_weights = (1 - probs) ** self.gamma 
        a = denom_weights * log_probs
        denominator = torch.clamp(-torch.sum(a, dim=1), min=self.small_value)
        focal_loss = (numerator / denominator).mean()
        # print("probs min/max/mean:", probs.min().item(), probs.max().item(), probs.mean().item())
        # print("target_probs min/max:", target_probs.min().item(), target_probs.max().item())
        # print("numerator any nan?", torch.isnan(numerator).any().item())
        # print("denominator any nan?", torch.isnan(denominator).any().item())
        return focal_loss
    
    def mae_loss(self, logits, targets):
        probs = torch.softmax(logits, dim=1) # Makes all elements between 0 and 1 such that you sum each element in a row it becomes 1
        one_hot_targets = F.one_hot(targets, num_classes=logits.shape[1]).float()
        #torch.nn.functional.one_hot(tensor, num_classes=-1)
        #tensor: A tensor containing class indices (integers).
        #num_classes: The total number of classes (optional, but required for consistent sizing).
        return torch.abs(probs - one_hot_targets).mean() #PyTorch’s .mean() without any arguments computes the mean of all elements in the tensor — across all dimensions.

    def __call__(self, logits, targets):
        active_loss = self.nfl_loss(logits, targets)
        passive_loss = self.mae_loss(logits, targets)
        loss = self.alpha * active_loss + self.beta * passive_loss
        # print(f'Active loss: {active_loss}, Passive loss: {passive_loss}, Loss: {loss}')
        return loss

class NCE_RCE:
    def __init__(self, alpha=0.5,beta=0.5,smooth=0.1,clip_value=1e-7):
        self.smooth = smooth
        self.alpha = alpha
        self.clip_value = clip_value
        self.beta = beta
        self.small_value=1e-4

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
        numerator = -prob_true
        denominator = -torch.sum(log_probs, dim=1)       

        loss = numerator/denominator
        return loss.mean()
    def rce_loss(self, logits, targets):
        probs = torch.softmax(logits, dim=1)  # p(k|x)
        batch_size, num_classes = probs.shape

        with torch.no_grad():
            q = torch.full_like(probs, fill_value=self.small_value)  # q(k|x) ≈ 1e-4
            q.scatter_(1, targets.view(-1, 1), 1.0)  # q(y|x) = 1
            q = torch.clamp(q, min=self.small_value, max=1.0)
        
        loss = -torch.sum(probs * torch.log(q), dim=1)
        return loss.mean()

    def __call__(self, logits, targets):
        active_loss = self.nce_loss(logits, targets)
        passive_loss = self.rce_loss(logits, targets)
        loss = self.alpha * active_loss + self.beta * passive_loss
        # print(f'Active loss: {active_loss}, Passive loss: {passive_loss}, Loss: {loss}')
        return loss
