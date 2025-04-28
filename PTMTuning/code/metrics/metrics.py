import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiNegTripletLoss(nn.Module):
    """
    Triplet loss for one positive and multiple negatives per anchor.
    Positive is contents[:,0,:], negatives are contents[:,1:,:].
    L = mean(max(d(a,p) - d(a,n_i) + margin, 0)) averaged over all negatives and batch.
    """
    def __init__(self, margin: float = 1.0, p: int = 2):
        super().__init__()
        self.margin = margin
        self.p = p

    def forward(self, anchor: torch.Tensor, contents: torch.Tensor) -> torch.Tensor:
        # anchor: [batch_size, hidden_size]
        # contents: [batch_size, num_contents, hidden_size]
        pos = contents[:, 0, :]              # [batch_size, hidden_size]
        negs = contents[:, 1:, :]            # [batch_size, num_negatives, hidden_size]

        # compute distances
        d_pos = torch.norm(anchor - pos, p=self.p, dim=1)                   # [batch_size]
        # expand anchor for negs
        a_exp = anchor.unsqueeze(1).expand_as(negs)                         # [batch_size, num_neg, hidden]
        d_negs = torch.norm(a_exp - negs, p=self.p, dim=2)                  # [batch_size, num_neg]

        # triplet losses per negative
        losses = F.relu(d_pos.unsqueeze(1) - d_negs + self.margin)          # [batch_size, num_neg]
        return losses.mean()   
    



class UnifiedCoTLoss(nn.Module):
    """
    Unified CoT Alignment & Triplet Loss supporting both multi-negative (and in-batch negatives,not support now).
    Usage:
      - Multi-negative scenario: call forward(anchor, contents=..., cot=...)
      - In-batch scenario: call forward(anchor, positive=..., cot=...)
    CoT loss: alpha*triplet + beta*||anchor-cot||^2 + gamma*||cot-positive||^2
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, margin=1.0, p=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin
        assert self.alpha+self.beta+self.gamma == 1.0, "alpha + beta + gamma must equal 1.0"
        self.p = p
        self.multi_neg = MultiNegTripletLoss(margin, p)


    def forward(self, anchor: torch.Tensor, cot: torch.Tensor, contents: torch.Tensor=None) -> torch.Tensor:
        # Compute triplet loss
        B, D = anchor.size(0), anchor.size(1)
        l_triplet = self.multi_neg(anchor, contents) if self.alpha > 0 else 0
        # CoT alignment and consistency
        l_align = torch.norm(anchor.unsqueeze(1) - cot.view(B, 2, D), p=self.p, dim=1).pow(2).mean()
        
        if self.gamma==0.0:
            l_consis = 0
        else:
            l_consis = torch.norm(cot.view(-1, 2, cot.size(1))  - contents[:, 0, :], p=self.p, dim=1).pow(2).mean()
        return self.alpha * l_triplet + self.beta * l_align + self.gamma * l_consis