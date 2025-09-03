# make generic imports 
import pyHarm
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import copy
import pyHarm.Systems.ABCSystem
## POST-TREATMENT FUNCTIONS THAT ARE USED IN THE TEST CASES
RATIO_FIG = (15,8)

def get_frf(M:pyHarm.Maestro,sub="m1",node=0,direction=0,deriv:int=0) :
    if M.system.adim : 
        lc, wc = M.system.lc, M.system.wc
    else :
        lc, wc = 1., 1. 
    index_kept = M.getIndex(sub,node,direction)
    SA = [sol for sol in M.nls["FRF"].SolList if sol.flag_accepted]
    om = np.array([sol.x[-1] * wc for sol in SA])
    nabla = M.system.LE[0].nabla 
    D = M.system.LE[0].D 
    nablo = np.linalg.matrix_power(nabla,deriv)
    amp = np.array([np.linalg.norm(sol.x[-1]**(deriv)*nablo@( M.system.get_full_disp(sol.x)[index_kept] * lc )) for k,sol in enumerate(SA)])
    return om, amp 


def export_sol(M:pyHarm.Maestro, path="./", prefix='') :
    if M.system.adim : 
        lc, wc = M.system.lc, M.system.wc
    else :
        lc, wc = 1., 1. 
    edf = M.system.expl_dofs
    # export to path : 
    edf.to_csv(os.path.join(path, f"{prefix}_edf.csv"))
    for k,v in M.nls.items() : 
        SA = [sol for sol in M.nls["FRF"].SolList if sol.flag_accepted]
        om = np.array([sol.x[-1] * wc for sol in SA]).reshape(1,-1)
        xx = np.concatenate([(M.system.get_full_disp(sol.x) * lc).reshape(-1,1) for k,sol in enumerate(SA)],axis=1)
        xxom = pd.DataFrame(np.concatenate([xx,om],axis=0))
        xxom.to_csv(os.path.join(path, f"{prefix}_{k}_sol.csv"))
        
def create_directory_relative(path):
    """
    Creates a directory using a relative path if it does not already exist.

    Parameters:
    path (str): The relative path of the directory to create.
    """
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' created or already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")


def prep_fig(fp=None, figsize=RATIO_FIG, logscale=False):
    fig,ax = plt.subplots(figsize=figsize)
    if isinstance(fp,np.ndarray) : 
        ax.set_xticks(fp)
        ax.set_xticklabels([f"{f:.2e}" for f in fp])
    ax.grid()
    if logscale : 
        ax.set_yscale("log")
    return fig,ax

def modify_input_dict(entry_dict, modification_dict) : 
    new_entry_dict = copy.deepcopy(entry_dict)
    for k,v in modification_dict.items() : 
        try : 
            new_entry_dict[k] = v
        except : 
            print(f"Warning : could not change the key {k} in the entry dict with new value {v}. Check that key is proper")
    return new_entry_dict

def mid(*args):
    return modify_input_dict(*args)

## some stuff for the plot : 
def build_style(*args) : 
    style_dict = dict()
    label = list()
    for a in args : 
        aa = copy.copy(a)
        label.append(aa.pop('label'))
        style_dict = style_dict | a
    style_dict['label'] = ' '.join(label)
    return style_dict


# This is provided by CHARLEUX as best EPSILON selection for the DLFT method !
# see : Numerical and Experimental Study of Friction Damping in Blade Attachments of Rotating Bladed Disks, (2006), International Journal of Rotating Machinery
def spectral_radius_dynamicMat(sys:pyHarm.Systems.ABCSystem, om: float, _nh:int|list[int]=3) :
    if isinstance(_nh,list):
        nh = max(_nh)
    else : nh = _nh
    xx0 = np.zeros(sys.ndofs)
    x0 = np.concatenate([xx0, np.array([om])])
    Z = sys._jacobian(sys.LE_linear, x0)[0]
    n_block = int((Z.shape[0]-1)/2.)
    Z00 = Z[0:n_block,0:n_block]
    Z11 = Z[n_block:n_block*3,n_block:n_block*3]
    Zhh = np.kron(np.eye(nh), Z11)
    Zextended = np.block(
        [
            [Z00,np.zeros((Z00.shape[0], Zhh.shape[1]))],
            [np.zeros((Zhh.shape[0], Z00.shape[1])),Zhh]
        ]
    )
    return np.max(np.abs(np.linalg.eig(Zextended)[0]))