import torch

class Similarity(torch.nn.Module):
    def __init__(self, normalise_dim=-1, tau=0.2, hard=False):
        super(Similarity, self).__init__()
        self.dim = normalise_dim
        self.tau = tau
        self.hard = hard

    def forward(self, log_alpha):
        log_alpha = log_alpha / self.tau
        alpha = torch.exp(log_alpha - (torch.logsumexp(log_alpha, dim=self.dim, keepdim=True)))

        if self.hard:
            # Straight through.
            index = alpha.max(self.dim, keepdim=True)[1]
            alpha_hard = torch.zeros_like(alpha, memory_format=torch.legacy_contiguous_format).scatter_(self.dim, index, 1.0)
            ret = alpha_hard - alpha.detach() + alpha
        else:
            ret = alpha
        return ret
    
def fsf(evecs1, evecs2, p2p, k_ini, k_end, step):
    p2p_zo=p2p
    sim=Similarity()
    evec1,evec2=evecs1[:,:k_ini], evecs2[:,:k_ini]
    C21=torch.linalg.pinv(evec1)@evec2[p2p_zo]
    similarity = torch.bmm(evec2[None], (evecs1@C21)[None].transpose(1, 2))
    P21 = sim(similarity)

    for neig in range(k_ini,k_end,step):
        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        C21=torch.linalg.pinv(evec1)@P21@evec2
        similarity = torch.bmm(evec2[None], (evecs1@C21)[None].transpose(1, 2))
        P21 = sim(similarity)

    C21=torch.linalg.pinv(evec1)@P21@evec2
    p2p_zo=nearest(evec1@C21,evec2)
    return p2p_zo
#### zoomout and variants    --> ONLY INTRINSIC
def zoomout(evecs1, evecs2, p2p, k_ini, k_end, step):
    p2p_zo=p2p
    for neig in range(k_ini,k_end,step):
        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        C21=torch.linalg.pinv(evec1)@evec2[p2p_zo]
        p2p_zo=nearest(evec1,evec2@C21.T)
        
    return p2p_zo

def ab_zoomout(evecs1, evecs2, p2p, k_ini, k_end, step):
    p2p_zo=p2p
    for neig in range(k_ini,k_end,step):
        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        C12=torch.linalg.pinv(evec2[p2p_zo])@evec1
        p2p_zo=nearest(evec1,evec2@C12)
    
    return p2p_zo

def ima(evecs1, evecs2, p2p, k_ini, k_end, step):
    p2p_zo=p2p
    for neig in range(k_ini,k_end,step):
        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        C21=torch.linalg.pinv(evec1)@evec2[p2p_zo]
        p2p_zo=nearest(evec1@C21,evec2)
     
    return p2p_zo

def abzo_lsq(evecs1, evecs2, p2p, k_ini, k_end, step):
    p2p_zo=p2p
    for neig in range(k_ini,k_end,step):
        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]

        C21=torch.linalg.lstsq(evec2[p2p_zo],evec1)
        p2p_zo=nearest(evec1,evec2@C21)
        
    return p2p_zo

