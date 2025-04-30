import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiPosNegTripletLoss(nn.Module):
    """
    Triplet loss for one anchor and multiple positives & negatives.
    positives: samples[:, :P, :]
    negatives: samples[:, P:, :]
    L = mean(  ReLU( d(a, pos_i) - d(a, neg_j) + margin )  ) 
      over all i in [0,P), j in [0,Nneg), batch
    """
    def __init__(self, margin: float = 1.0, p: int = 2):
        super().__init__()
        self.margin = margin
        self.p = p

    def forward(self, anchor: torch.Tensor, samples: torch.Tensor, n_pos: int) -> torch.Tensor:
        # anchor: [B, D]
        # samples: [B, P + Nneg, D]
        B, total, D = samples.shape
        P = n_pos
        pos  = samples[:, :P, :]       # [B, P, D]
        negs = samples[:, P:, :]       # [B, Nneg, D]

        # 计算 anchor→每个正例的距离 [B, P]
        a_exp_pos = anchor.unsqueeze(1).expand(B, P, D)  
        d_pos     = torch.norm(a_exp_pos - pos, p=self.p, dim=2)  # [B, P]

        # 计算 anchor→每个负例的距离 [B, Nneg]
        a_exp_neg = anchor.unsqueeze(1).expand_as(negs)          
        d_negs    = torch.norm(a_exp_neg - negs, p=self.p, dim=2) # [B, Nneg]

        # 对每个 (pos_i, neg_j) 计算 triplet loss 并平均
        # Shape 扩成 [B, P, Nneg]
        d_pos = d_pos.unsqueeze(2)    # [B, P, 1]
        d_negs = d_negs.unsqueeze(1)  # [B, 1, Nneg]
        losses = F.relu(d_pos - d_negs + self.margin)  # [B, P, Nneg]
        return losses.mean()
    



class UnifiedCoTLoss(nn.Module):
    """
    α * Triplet(q, c⁺, c⁻)
  + β * CoT-align (支持多正例 CoT + 多负例 Cot)
  + γ * CoT→Content-consistency

  前面一共传入：
    anchor:   [B, D]
    cot:      [B * (Pcot + Ncot), D]
    contents: [B * (Pcont + Ncont), D]  # 与原版一致

  调用时指定：
    Pcot：CoT 中的正例个数
    Pcont：contents 中的正例个数（原先默认为1）
    Ncot, Ncont 则由总条数减去正例数得到
    """
    def __init__(self, alpha=0.3, beta=0.3, gamma=0.4, margin=1.0, p=2):
        super().__init__()
        assert abs(alpha + beta + gamma - 1.0) < 1e-6
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.p      = p
        self.margin = margin
        self.triplet = MultiPosNegTripletLoss(margin, p)

    def forward(
        self,
        anchor:   torch.Tensor,      # [B, D]
        cot:      torch.Tensor,      # [B*(Pcot+Ncot), D]
        contents: torch.Tensor = None,# [B*(Pcont+Ncont), D]

    ) -> torch.Tensor:
        # Pcot = 2         # CoT 中正例数量
        # Pcont = 1            # Content 中正例数量
        B, D = anchor.size()

        # —— Q→CoT 部分 —— 
        cot_all = cot.view(B, -1, D)            # [B, Pcot+Ncot, D]
        Pcot    = min(2, cot_all.size(1))
        if cot_all.size(1) > Pcot and self.beta > 0:
            # 多于正例 + 负例，就做 Triplet
            l_qcot = self.triplet(anchor, cot_all, n_pos=Pcot)
        else:
            # 只有正例时，L2 回归：用所有正例平均向量
            pos_cot = cot_all[:, :Pcot, :].mean(dim=1)  # [B, D]
            l_qcot  = torch.norm(anchor - pos_cot, p=self.p, dim=1).pow(2).mean()

        
        contents = contents.view(B, -1, D)
        Pcont = min(1,contents.size(1))
        if  self.gamma > 0:
            # —— Content:1→CoT:n 一致性 ——
            l_cotc = self.triplet(contents[:,0,:], cot_all, n_pos=Pcot)
            # —— 1→CoT→Content:n 一致性 ——
            if contents.size(1)>Pcont:
                l_cotc += self.triplet(cot_all[:,0,:], contents, n_pos=Pcont)
                l_cotc += self.triplet(cot_all[:,1,:], contents, n_pos=Pcont)
                l_cotc = l_cotc/3
        else:
            l_cotc = 0.0

        # —— Q→Content 三元组 —— 
        if contents.size(1)>Pcont and self.alpha > 0:
            l_qc = self.triplet(anchor, contents, n_pos=1)
        else:
            l_qc = 0.0

        # —— 加权合并 —— 
        return self.alpha * l_qc + self.beta * l_qcot + self.gamma * l_cotc