'''
This file contains the implementation of the linear refinement methods. In particualr, we implement the following methods:
-  Zoomout
-  Iterative Meta Algorithm
-  Fast Sinkhorn Filters
-  Adjoint Bijective Zoomout
-  ZoomOut with mass (for non orthogonal basis)
-  Iterative Meta Algorithm with mass

'''
import torch

#NOTE: In the original implementation the following import is used:
#from torch_cluster import nearest
from nam_utils.sinkhorn_utils import Sinkhorn
import scipy

def nearest(src, dst):
    dist = torch.cdist(src, dst)
    return torch.argmin(dist, dim=1)

def fsf(evecs1, evecs2, p2p, k_ini=20, k_end=200, step=20 ,mass_inv_sqrt_input=None,mass_inv_sqrt_input2=None):
    if mass_inv_sqrt_input==None:
        mass_inv_sqrt_input=torch.eye(evecs1.shape[0]).to(evecs1.device)
    if mass_inv_sqrt_input2==None:
        mass_inv_sqrt_input2=torch.eye(evecs2.shape[0]).to(evecs2.device)

    p2p_zo=p2p
    mass_inv_sqrt=mass_inv_sqrt_input[:k_ini,:k_ini]
    mass_inv_sqrt2=mass_inv_sqrt_input2[:k_ini,:k_ini]

    sim=Sinkhorn(lambda_sink=0.2,num_sink=5).to(evecs1.device)
    evec1,evec2=evecs1[:,:k_ini], evecs2[:,:k_ini]
    C21=torch.linalg.pinv(evec1)@evec2[p2p_zo]
    similarity = torch.bmm( (evec1@C21@mass_inv_sqrt)[None],(evec2[None]@mass_inv_sqrt2).transpose(1, 2))
    P21 = sim(similarity)

    for neig in range(k_ini,k_end,step):
        mass_inv_sqrt=mass_inv_sqrt_input[:neig,:neig]
        mass_inv_sqrt2=mass_inv_sqrt_input2[:neig,:neig]

        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        C21=torch.linalg.pinv(evec1)@P21[0]@evec2
        similarity = torch.bmm((evec1@C21@mass_inv_sqrt)[None],(evec2[None]@mass_inv_sqrt2).transpose(1, 2))
        P21 = sim(similarity)

    C21=torch.linalg.pinv(evec1)@P21[0]@evec2
    p2p_zo=nearest(evec1@C21@mass_inv_sqrt,evec2@mass_inv_sqrt2)
    return p2p_zo


def ab_zoomout(evecs1, evecs2, p2p, k_ini=20, k_end=200, step=20 ,mass_inv_sqrt_input=None,mass_inv_sqrt_input2=None):
    if mass_inv_sqrt_input==None:
        mass_inv_sqrt_input=torch.eye(evecs1.shape[0]).to(evecs1.device)
    if mass_inv_sqrt_input2==None:
        mass_inv_sqrt_input2=torch.eye(evecs2.shape[0]).to(evecs2.device)
    p2p_zo=p2p
    for neig in range(k_ini,k_end,step):
        mass_inv_sqrt=mass_inv_sqrt_input[:neig,:neig]
        mass_inv_sqrt2=mass_inv_sqrt_input2[:neig,:neig]

        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        C12=torch.linalg.pinv(evec2[p2p_zo])@evec1
        p2p_zo=nearest(evec1@mass_inv_sqrt,evec2@C12@mass_inv_sqrt2)
    
    return p2p_zo

def ima(evecs1, evecs2, p2p, k_ini=20, k_end=200, step=20 ,mass_inv_sqrt_input=None,mass_inv_sqrt_input2=None):
    if mass_inv_sqrt_input==None:
        mass_inv_sqrt_input=torch.eye(evecs1.shape[0]).to(evecs1.device)
    if mass_inv_sqrt_input2==None:
        mass_inv_sqrt_input2=torch.eye(evecs2.shape[0]).to(evecs2.device)
    p2p_zo=p2p
    for neig in range(k_ini,k_end,step):
        mass_inv_sqrt=mass_inv_sqrt_input[:neig,:neig]
        mass_inv_sqrt2=mass_inv_sqrt_input2[:neig,:neig]


        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        C21=torch.linalg.pinv(evec1)@evec2[p2p_zo]
        p2p_zo=nearest(evec1@C21@mass_inv_sqrt,evec2@mass_inv_sqrt2)
     
    return p2p_zo




#### zoomout and variants    --> ONLY INTRINSIC
def zoomout(evecs1, evecs2, p2p, k_ini=20, k_end=200, step=20):
    p2p_zo=p2p
    for neig in range(k_ini,k_end,step):
        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        C21=torch.linalg.pinv(evec1)@evec2[p2p_zo]
        p2p_zo=nearest(evec1,evec2@C21.T)
        
    return p2p_zo

#### zoomout and variants    --> ONLY INTRINSIC
def zoomout_mass(evecs1, evecs2, p2p, k_ini=20, k_end=200, step=20 ,mass_sp1=None,mass_sp2=None):
    if mass_sp1==None:
        mass_sp1=torch.eye(evecs1.shape[0]).to(evecs1.device)
    if mass_sp2==None:
        mass_sp2=torch.eye(evecs2.shape[0]).to(evecs2.device)
    p2p_zo=p2p
    for neig in range(k_ini,k_end,step):
        mass_x=mass_sp1[:neig,:neig]
        mass_y=mass_sp2[:neig,:neig]

        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        C21=torch.linalg.pinv(evec1)@evec2[p2p_zo]
        A=torch.linalg.inv(mass_y)@C21.T@mass_x
        mass_inv_sqrt=torch.tensor(scipy.linalg.sqrtm(torch.linalg.inv((mass_x)).cpu().numpy())).to(evecs1.device)
        print(mass_inv_sqrt.dtype)
        p2p_zo=nearest(evec1@mass_inv_sqrt,evec2@A@mass_inv_sqrt)
        
    return p2p_zo
    

def abzo(evecs1, evecs2, p2p, k_ini=20, k_end=200, step=20):
    p2p_zo=p2p
    for neig in range(k_ini,k_end,step):

        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        C12=torch.linalg.pinv(evec2[p2p_zo])@evec1
        p2p_zo=nearest(evec1,evec2@C12)
    
    return p2p_zo



def abzo_mass(evecs1, evecs2, p2p, k_ini=20, k_end=200, step=20 ,mass_sp1=None,mass_sp2=None):
    if mass_sp1==None:
        mass_sp1=torch.eye(evecs1.shape[0]).to(evecs1.device)
    if mass_sp2==None:
        mass_sp2=torch.eye(evecs2.shape[0]).to(evecs2.device)
    p2p_zo=p2p
    for neig in range(k_ini,k_end,step):
        mass_x=mass_sp1[:neig,:neig]
        mass_y=mass_sp2[:neig,:neig]
        evec1,evec2=evecs1[:,:neig], evecs2[:,:neig]
        A=torch.linalg.pinv(evec2[p2p_zo])@evec1
        mass_inv_sqrt=torch.tensor(scipy.linalg.sqrtm(torch.linalg.inv((mass_x)).cpu().numpy())).to(evecs1.device)
        p2p_zo=nearest(evec1@mass_inv_sqrt,evec2@A@mass_inv_sqrt)
        
    return p2p_zo
