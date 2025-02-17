import torch
from torch_cluster import nearest

from model.neural_adjoint_map import *
from model.optimizer import *
from ..utils.sinkhorn_utils import Sinkhorn

    

def get_nam(emb1,emb2):
    nam12=Neural_Adjoint_Map(emb1.shape[1]).to(emb1.device)
    loss_handler=LossHandler()
    opt=NAMOptimizer(nam12,loss_handler)
    opt.optimize(emb1,emb2)
    return nam12



def neural_ima(evecs1, evecs2, p2p, k_ini, k_end, step):
    p2p_zo=p2p
    for neig in range(k_ini,k_end,step):
        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        nam21=get_nam(evec2[p2p_zo],evec1)
        p2p_zo=nearest(nam21(evec1),evec2)
     
    return p2p_zo

def neural_zoomout(evecs1, evecs2, p2p, k_ini, k_end, step):
    p2p_zo=p2p
    for neig in range(k_ini,k_end,step):
        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        nam21=get_nam(evec1,evec2[p2p_zo])
        p2p_zo=nearest(evec1,nam21(evec2))
        
    return p2p_zo



def neural_fsf(evecs1, evecs2, p2p, k_ini, k_end, step):
    p2p_zo=p2p
    sim=Sinkhorn(lambda_sink=0.2,num_sink=5)
    evec1,evec2=evecs1[:,:k_ini], evecs2[:,:k_ini]
    nam21=get_nam(evec2[p2p_zo],evec1)
    similarity = torch.bmm((nam21(evec1))[None], evec2[None].transpose(1, 2))
    P21 = sim(similarity).detach()

    for neig in range(k_ini,k_end,step):
        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        nam21=get_nam(P21[0]@evec2,evec1)
        similarity = torch.bmm((nam21(evec1))[None], evec2[None].transpose(1, 2))
        P21 = sim(similarity).detach()

    nam21=get_nam(evec2[p2p_zo],evec1)
    p2p_zo=nearest(nam21(evec1),evec2)
    return p2p_zo

