import numpy as np
from collections import namedtuple
import torch
import math
from typing import Optional
import os
import pickle as pkl

def circshift_numpy(matrix, nb):
    if matrix.shape[0] > 1: 
        return np.roll(matrix, nb, axis=0)
    return np.roll(matrix, nb, axis=1)

# def circshift_torch(matrix, nb):
#     if matrix.size()[0] > 1: 
#         return torch.roll(matrix, nb, dims=0)
#     return torch.roll(matrix, nb, dims=1)

def circshift_torch(matrix, nb,device=None):
    if matrix.size()[0] > 1: 
        return roll(matrix, nb, dim=0)
    return roll(matrix, nb, dim=1)

def roll(x: torch.Tensor, shift: int, dim: int = -1, fill_pad: Optional[int] = None):

    if 0 == shift:
        return x

    elif shift < 0:
        shift = -shift
        gap = x.index_select(dim, torch.arange(shift,device=x.device))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([x.index_select(dim, torch.arange(shift, x.size(dim),device=x.device)), gap], dim=dim)

    else:
        shift = x.size(dim) - shift
        gap = x.index_select(dim, torch.arange(shift, x.size(dim),device=x.device))
        if fill_pad is not None:
            gap = fill_pad * torch.ones_like(gap, device=x.device)
        return torch.cat([gap, x.index_select(dim, torch.arange(shift,device=x.device))], dim=dim)


def _cut_to_parts_n(in_list,n_parts=2):
    part_len = math.ceil(len(in_list)/n_parts)
    for i in range(n_parts):
        yield in_list[i*part_len:(i+1)*part_len if i+part_len<len(in_list) else len(in_list)]
        
def cut_to_parts(in_list,n_parts=None,batch_size=5):
    if n_parts is not None:
        for res in _cut_to_parts_n(in_list,n_parts):
            yield res
    
    else:
        start = 0
        end = min(batch_size,len(in_list))
        
        yield in_list[start:end]
        while end!=len(in_list):
            start = start+batch_size
            end = min(end+batch_size,len(in_list))
            yield in_list[start:end]
        
def save_args(basedir='.',prefix='_',suffix='.pkl',**kwargs):
    os.system('mkdir -p '+basedir)
    
    for argname,arg in kwargs.items():
        fname = basedir+os.sep+prefix+argname+suffix
        try:
            argtype = arg.type()
            if 'torch' in argtype and 'Tensor' in argtype:
                torch.save(arg,fname)
        except AttributeError as e:
            with open(fname,'wb') as f:
                pkl.dump(arg,f)
                
def _open_pickled(filepath):
    try:
        tens = torch.load(filepath)
        return tens
           
    except Exception as e:
        with open(filepath,'rb') as f:
            return pkl.load(f)
    
       
        
def load_args(dir_to_load):
    files = os.listdir(dir_to_load)
    prefixes = set([f.split('__')[0] for f in files])
    for prefix in prefixes:
        prefix_files = sorted([f for f in files if prefix==f.split('__')[0]])
        yield [_open_pickled(dir_to_load+os.sep+f) for f in prefix_files]
        
def load_kwargs(dir_to_load):
    files = os.listdir(dir_to_load)
    prefixes = set([f.split('__')[0] for f in files])
    for prefix in prefixes:
        prefix_files = sorted([f for f in files if prefix==f.split('__')[0]])
        yield dict([(f.split('__')[1].split('.')[0],_open_pickled(dir_to_load+os.sep+f)) for f in prefix_files])
    

Scenario = namedtuple('Scenario', [
    'SNR', 'seed','index', 'RB_num', 'N_TTI', 'UE_indx', 'UE_number', 
    'N_seeds', 'N_scenarios', 'N_pilot', 'RB_size', 'Nrx', 'upsample_factor', 'Nfft', 'N_shift', 'N_response', 'comb'])