'''
This file contains the implementation of the refinements based on the Neural Adjoint Map representation. In particualr, we implement the following methods:
-  Neural Iterative Meta Algorithm
-  Neural ZoomOut
-  Neural Fast Sinkhorn Filters

'''
import torch
#NOTE: In the original implementation the following import is used:
#from torch_cluster import nearest
import scipy

from model.neural_adjoint_map import *
from model.optimizer import *
from nam_utils.sinkhorn_utils import Sinkhorn

def nearest(src, dst):
    dist = torch.cdist(src, dst)
    return torch.argmin(dist, dim=1)


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

def alternate_neural_zoomout(evecs1, evecs2,f1,f2, p2p, k_ini, k_end, step):
    
    p2p_zo=p2p
    for neig in range(k_ini,k_end,step):
        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        nam21=get_nam(evec1,evec2[p2p_zo])
        p2p_zo=nearest(evec1,nam21(evec2))
        nam21=get_nam(f1, f2[p2p_zo])
        p2p_zo=nearest(f1,nam21(f2))
        
    return p2p_zo


def neural_zoomout_mass(evecs1, evecs2, p2p, k_ini, k_end, step,mass_sp1=None,mass_sp2=None):
    if mass_sp1==None:
        mass_sp1=torch.eye(evecs1.shape[0]).to(evecs1.device)
    if mass_sp2==None:
        mass_sp2=torch.eye(evecs2.shape[0]).to(evecs2.device)

    p2p_zo=p2p
    for neig in range(k_ini,k_end,step):
        mass_x=mass_sp1[:neig,:neig]
        mass_y=mass_sp2[:neig,:neig]

        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        nam21=get_nam(evec1,evec2[p2p_zo])
        mass_inv_sqrt=torch.tensor(scipy.linalg.sqrtm(torch.linalg.inv((mass_x)).cpu().numpy())).to(evecs1.device)
        p2p_zo=nearest(evec1@mass_inv_sqrt,nam21(evec2)@mass_inv_sqrt)
        
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

