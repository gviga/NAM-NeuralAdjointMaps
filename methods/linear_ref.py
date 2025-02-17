import torch
from torch_cluster import nearest
from ..utils.sinkhorn_utils import Sinkhorn

def fsf(evecs1, evecs2, p2p, k_ini, k_end, step):
    p2p_zo=p2p
    sim=Sinkhorn(lambda_sink=0.2,num_sink=5)
    evec1,evec2=evecs1[:,:k_ini], evecs2[:,:k_ini]
    C21=torch.linalg.pinv(evec1)@evec2[p2p_zo]
    similarity = torch.bmm( (evec1@C21)[None],evec2[None].transpose(1, 2))
    P21 = sim(similarity)

    for neig in range(k_ini,k_end,step):
        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        C21=torch.linalg.pinv(evec1)@P21[0]@evec2
        similarity = torch.bmm((evec1@C21)[None],evec2[None].transpose(1, 2))
        P21 = sim(similarity)

    C21=torch.linalg.pinv(evec1)@P21[0]@evec2
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

