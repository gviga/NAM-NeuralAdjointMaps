import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class NCESoftmaxLoss(nn.Module):
    def __init__(self, nce_t=0.07, nce_num_pairs=1024):
        super().__init__()
        self.nce_t = nce_t
        self.nce_num_pairs = nce_num_pairs
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, features_1, features_2, map21):
        """Computes the NCE loss between two sets of features.

        Args:
            features_1 (torch.Tensor): The first set of features. Shape (B, N1, F).
            features_2 (torch.Tensor): The second set of features. Shape (B, N2, F).
            map21 (list): p2p correspondences between the two sets of features. Shape (B, N2).

        Returns:
            torch.Tensor: The NCE loss.
        """
        map21 = map21.view(features_1.size(0), -1)
        loss = 0

        for i in range(features_1.size(0)):
            map_21, feat1, feat2 = map21[i], features_1[i], features_2[i]
            mask = map_21 != -1
            map_21_masked = map_21[mask]

            if map_21_masked.shape[0] > self.nce_num_pairs:
                selected = np.random.choice(map_21_masked.shape[0], self.nce_num_pairs, replace=False)
            else:
                selected = torch.arange(map_21_masked.shape[0])

            query = feat1[map_21_masked[selected]]
            keys = feat2[mask][selected]

            logits = - torch.cdist(query, keys)
            logits = torch.div(logits, self.nce_t)
            labels = torch.arange(selected.shape[0]).long().to(feat1.device)
            loss += self.cross_entropy(logits, labels)

        return loss


#jacobian losses
def compute_jacobian_determinant(model, x):
    x.requires_grad_(True)
    y,_,_ = model(x)
    jacobian = []
    for i in range(y.size(1)):
        grad_output = torch.zeros_like(y)
        grad_output[:, i] = 1
        jacobian.append(torch.autograd.grad(y, x, grad_output, retain_graph=True, create_graph=True)[0])
    jacobian = torch.stack(jacobian, dim=2)
    det = torch.det(jacobian)
    return det



def cdot(X, Y, dim):
    assert X.dim() == Y.dim()
    return torch.sum(torch.mul(X, Y), dim=dim)

class DirichletLoss(nn.Module):
    def __init__(self, normalize=False, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize

    def forward(self, feats, L):
        #assert feats.dim() == 3

        if self.normalize:
            feats = F.normalize(feats, p=2, dim=-1)

        de = cdot(feats, torch.bmm(L, feats), dim=1)
        loss = torch.mean(de)

        return self.loss_weight * loss


class SpectralDirichletLoss(nn.Module):
    def __init__(self, normalize=False, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize

    def forward(self, feats, evals):
        #feats has to be projected 

        if self.normalize:
            feats = F.normalize(feats, p=2, dim=-1)

        de =    cdot(feats, torch.bmm(torch.diag(evals)[None], feats), dim=1)
        loss = torch.mean(de)

        return self.loss_weight * loss
