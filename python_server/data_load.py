from functools import lru_cache
from scipy.io import loadmat
import torch
import numpy as np


@lru_cache(maxsize=300)
def cached_loadmat(fname):
    return loadmat(fname)

def data_load(scen, path, astype=torch.float64, device='cpu', n_pilots=1):
    comb = scen.comb
    if n_pilots==1:
        # N-of-pilot-symbols
        N_pilot_sym = scen.N_pilot*scen.N_TTI
        N_data_sym = (14-N_pilot_sym)*scen.N_TTI;
        N_used = scen.RB_num*scen.RB_size
        pilot_positions=3
        data_positions1=np.array(range(0,3))
        data_positions2=np.array(range(4,14))
        #path_to_channel_mat = 'one_pilot_data/temp_chan_seed'+str(int(scenario.index))+'.mat'#str(int(scenario.index))
        H_tmp = cached_loadmat(path)
        try:
            H_tmp = H_tmp ['H_new']
        except KeyError as e:
            H_tmp = H_tmp ['Hfr'][0][0]

        # Extract pilot symbols
        h_pilot = torch.zeros(scen.Nrx, N_used, 2, dtype = astype, device=device)
        h_pilot[:,:,0] = torch.sqrt(torch.tensor(1+comb, dtype=astype))*torch.tensor(H_tmp[0,:,0:N_used:(1+comb), pilot_positions].real, dtype = astype, device=device)
        h_pilot[:,:,1] = torch.sqrt(torch.tensor(1+comb, dtype=astype))*torch.tensor(H_tmp[0,:,0:N_used:(1+comb), pilot_positions].imag, dtype = astype, device=device)

        # Extract data symbols
        h_data = torch.zeros(scen.Nrx, N_used, N_data_sym, 2, dtype = astype, device=device)
        for x in data_positions1:
            h_data[:,:,x,0] = torch.tensor(H_tmp[0,:,0:N_used, x].real, dtype = astype, device=device)
            h_data[:,:,x,1] = torch.tensor(H_tmp[0,:,0:N_used, x].imag, dtype = astype, device=device)
        for x in data_positions2:
            h_data[:,:,x-1,0] = torch.tensor(H_tmp[0,:,0:N_used, x].real, dtype = astype, device=device)
            h_data[:,:,x-1,1] = torch.tensor(H_tmp[0,:,0:N_used, x].imag, dtype = astype, device=device)
        return h_pilot, h_data
    elif n_pilots==2:
        N_pilot_sym = 2
        N_data_sym = 14 - N_pilot_sym
        N_used = scen.RB_num * scen.RB_size
        pilot_positions = [3, 10]
        data_positions1 = np.array(range(0, 3))
        data_positions2 = np.array(range(4, 10))
        data_positions3 = np.array(range(11, 14))

        H_tmp = cached_loadmat(path)
        try:
            H_tmp = H_tmp ['H_new']
        except KeyError as e:
            H_tmp = H_tmp ['Hfr'][0][0]

        # Extract pilot symbols
        h_pilot = torch.zeros(scen.Nrx, N_used, 4, device=device, dtype=astype)
        h_pilot[:, :, 0] = torch.sqrt(torch.tensor(1+comb, dtype=astype))*torch.tensor(H_tmp[0, :, 0:N_used:(1+comb), pilot_positions[0]].real, device=device, dtype=astype)
        h_pilot[:, :, 1] = torch.sqrt(torch.tensor(1+comb, dtype=astype))*torch.tensor(H_tmp[0, :, 0:N_used:(1+comb), pilot_positions[0]].imag, device=device, dtype=astype)
        h_pilot[:, :, 2] = torch.sqrt(torch.tensor(1+comb, dtype=astype))*torch.tensor(H_tmp[0, :, 0:N_used:(1+comb), pilot_positions[1]].real, device=device, dtype=astype)
        h_pilot[:, :, 3] = torch.sqrt(torch.tensor(1+comb, dtype=astype))*torch.tensor(H_tmp[0, :, 0:N_used:(1+comb), pilot_positions[1]].imag, device=device, dtype=astype)

        # Extract data symbols
        h_data = torch.zeros(scen.Nrx, N_used, N_data_sym, 2, dtype=astype,device=device)
        for x in data_positions1:
            h_data[:, :, x, 0] = torch.tensor(H_tmp[0, :, 0:N_used, x].real, device=device, dtype=astype)
            h_data[:, :, x, 1] = torch.tensor(H_tmp[0, :, 0:N_used, x].imag, device=device, dtype=astype)
        for x in data_positions2:
            h_data[:, :, x - 1, 0] = torch.tensor(H_tmp[0, :, 0:N_used, x].real, device=device, dtype=astype)
            h_data[:, :, x - 1, 1] = torch.tensor(H_tmp[0, :, 0:N_used, x].imag, device=device, dtype=astype)
        for x in data_positions3:
            h_data[:, :, x - 2, 0] = torch.tensor(H_tmp[0, :, 0:N_used, x].real, device=device, dtype=astype)
            h_data[:, :, x - 2, 1] = torch.tensor(H_tmp[0, :, 0:N_used, x].imag, device=device, dtype=astype)

        # s_tx = torch.ones(2, dtype=torch.float64) / torch.sqrt(torch.tensor([2, 2], dtype=torch.float64))
        #
        # X_data = torch.zeros(scen.Nrx, N_used, N_data_sym, 2, dtype=astype)
        # X_data[:, :, :, 0] = h_data[:, :, :, 0] * s_tx[0] - h_data[:, :, :, 1] * s_tx[1]
        # X_data[:, :, :, 1] = h_data[:, :, :, 0] * s_tx[1] + h_data[:, :, :, 1] * s_tx[0]

        return h_pilot, h_data

