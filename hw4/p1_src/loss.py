import torch
from torch.nn import functional as F

def euclidean_dis(x, y):
    """
        Parameters:
            x: (C x D)
            y: (N x D)
        Return:
            dis: (N x C)
    """
    C, N, D = x.shape[0], y.shape[0], x.shape[1]
    x = x.unsqueeze(0).expand(N,C,D)
    y = y.unsqueeze(1).expand(N,C,D)
    return torch.pow(x-y, 2).sum(2)

def cosine_similarity(x, y):
    """
        Parameters:
            x: (C x D)
            y: (N x D)
        Return:
            dis: (N x C)
    """
    C, N, D = x.shape[0], y.shape[0], x.shape[1]
    x = x.unsqueeze(0).expand(N,C,D)
    y = y.unsqueeze(1).expand(N,C,D)
    return F.cosine_similarity(x,y,dim=2)

def parametric_dis(x, y, para_dis):
    """
        Parameters:
            x: (C x D)
            y: (N x D)
            M: (D x D)
        Return:
            dis: (N x C)
    """
    def batch_dot_product(x, y):
        n = x.shape[0]
        dim = x.shape[1]
        return torch.bmm(x.view(n,1,dim), y.view(n,dim,1)).squeeze()
    eps = 1e-8  # In order to avoid runtime error
    C, N, D = x.shape[0], y.shape[0], x.shape[1]
    x = x.unsqueeze(0).expand(N,C,D).reshape(-1,D)
    y = y.unsqueeze(1).expand(N,C,D).reshape(-1,D)
    aug_x = para_dis(x) # (NxC, D)
    return (batch_dot_product(aug_x, y) / torch.clip(torch.norm(x,dim=1)*torch.norm(y,dim=1), min=eps)).view(N, C)

def cal_loss_and_acc(pred_emb, y, hp, device, dis_method, para_dis=None):
    """
        Description:
            Calculate loss and accuracy
        Parameters:
            pred_emb: (B, D) 
            y: (B)
    """
    classes = torch.unique(y) 
    cls_idx = [torch.nonzero(y == c) for c in classes]
    support_idx = [idx[:hp.k_shot] for idx in cls_idx]
    query_idx = [idx[hp.k_shot:] for idx in cls_idx]
    prototypes = [pred_emb[idx,:].mean(dim=0).squeeze(0) for idx in support_idx]
    prototypes = torch.stack(prototypes)   # C x D 
    query_emb = [pred_emb[idx,:].squeeze(0) for idx in query_idx]
    query_emb = torch.stack(query_emb).view(-1, prototypes.shape[-1])   # N x D
    if dis_method == 'euc':
        dist = euclidean_dis(prototypes, query_emb)   # N x C
    elif dis_method == 'cos':
        dist = -cosine_similarity(prototypes, query_emb) # N x C
    elif dis_method == 'para':
        dist = -parametric_dis(prototypes, query_emb, para_dis) # N x C
    prob = F.log_softmax(-dist, dim=1)  
    label = torch.arange(0, hp.n_way, dtype=torch.long).view(hp.n_way, 1).expand(hp.n_way, hp.n_query).reshape(-1).to(device)
    loss = F.cross_entropy(-dist, label)
    pred = prob.argmax(dim=-1)
    acc = (pred == label).sum() / len(pred)
    return loss, acc 