#!/usr/bin/env python
# coding: utf-8

## needed modules
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
import torch
import os,sys
from itertools import cycle
import pickle as pkl

# our own modules
module = sys.modules[__name__]
MODULE_ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(MODULE_ROOT_PATH+os.sep+'..')

from data_load import data_load
from helpers import circshift_torch
from helpers import save_args
from helpers import Scenario
import random

USE_WRITER=False

### this was recover_h
def CE_TTI(scen, h_f, ml, max_iter,device=None,dtype=torch.float32):
    N_used = scen.RB_num * scen.RB_size
    if len(h_f.shape) == 4:
        h_f = h_f.mean(dim=2)

    h_f_rec = 0 * h_f
    h_f_recover_data = torch.zeros(scen.Nrx, N_used, 14 - scen.N_pilot, 2, dtype=torch.float32, device=device)


    if USE_WRITER:
        writer = SummaryWriter()

    for it in range(max_iter):
        if USE_WRITER:
            writer.add_scalar('data/CE_TTI/iteration',it,it)
            writer.add_scalar('data/CE_TTI/SNR', scen.SNR, it)
        additional_data = {'current_iter': it}
        h_recovered_current = extract_peak_simple(scen, h_f, ml, ml_version=ml_version,
                                                  additional_data=additional_data,device=device,dtype=dtype)
        h_f = h_f - h_recovered_current
        h_f_rec = h_f_rec + h_recovered_current

        if USE_WRITER:
            writer.add_scalar('data/CE_TTI/h_f_power',torch.mean(h_f[:,:,0]**2+h_f[:,:,1]**2).cpu().data.numpy(),it)
            writer.add_scalar('data/CE_TTI/h_f_recovered_power',
                              torch.mean(h_f_rec[:, :, 0] ** 2 + h_f_rec[:, :, 1] ** 2).cpu().data.numpy(), it)

    if scen.comb==0:
        h_f_recovered = h_f_rec
    else:
        h_f_recovered = torch.zeros_like(h_f_rec)
        h_f_recovered[:,0:N_used:2, :] = h_f_rec[:, 0:N_used:2, :]/torch.sqrt(torch.tensor(2.))
        h_f_recovered[:,1:N_used-1:2, :] =  (h_f_rec[:, 0:N_used-2:2, :]+h_f_rec[:, 2:N_used:2, :])/torch.sqrt(torch.tensor(8.))
        h_f_recovered[:,-1, :] = h_f_rec[:,-2, :]/torch.sqrt(torch.tensor(2.))

    for j in range(scen.Nrx):

        h_f_recover_data[j, :, :, :] = h_f_recovered[j, :, :].clone().repeat(14-scen.N_pilot,1,1).permute(1,0,2)

    if USE_WRITER:
        writer.export_scalars_to_json("./tensorboard/all_scalars.json")
        writer.close()
    return h_f_recovered,h_f_recover_data


def noise_estim(scen, h_t_noisy):
    XZ = scen.upsample_factor * scen.Nfft
    length = int(XZ * ((1. / 2 - 1. / 16) - (1. / 4 + 1. / 16))) + int(XZ * ((1. - 1. / 16) - (3. / 4 + 1. / 16)))
    x = int(XZ * ((1. / 2 - 1. / 16) - (1. / 4 + 1. / 16)))
    y = int(XZ * (1. / 4 + 1. / 16))
    w = int(XZ * (3. / 4 + 1. / 16))
    z = int(XZ * (1. - 1. / 16))

    assert (len(h_t_noisy.shape) == 3 or len(h_t_noisy.shape) == 4)
    if len(h_t_noisy.shape) == 3:
        noise_array = torch.zeros((scen.Nrx, length, 2))
        noise_array[:, :x, :] = h_t_noisy[:, y:x + y]
        noise_array[:, x:, :] = h_t_noisy[:, w:z]
        noise_tmp = noise_array[:, :, 0] ** 2 + noise_array[:, :, 1] ** 2
    elif len(h_t_noisy.shape) == 4:
        noise_tmp = torch.zeros((scen.Nrx, length))
        noise_tmp[:, :x] = 2 * ((h_t_noisy ** 2)[:, y:x + y]).mean(dim=(2, 3))
        noise_tmp[:, x:] = 2 * ((h_t_noisy ** 2)[:, w:z]).mean(dim=(2, 3))

    noise_per_sc = torch.sum(noise_tmp) / (scen.N_pilot * 4.)  # noise per subcarrier
    NS2 = noise_per_sc / XZ
    noise_per_sample1 = torch.sum(torch.sum(noise_tmp[0:scen.Nrx:2, :], dim=0)) / (XZ / 4.)
    noise_per_sample2 = torch.sum(torch.sum(noise_tmp[1:scen.Nrx:2, :], dim=0)) / (XZ / 4.)
    noise_per_sample = noise_per_sample1 + noise_per_sample2
    return noise_per_sample, NS2

def SNR_estim(h_t_noisy,
              scen,
              clamp=-30 # min value
              ):
    ''' Estimate SNR'''
    XZ = scen.upsample_factor*scen.Nfft
    noise_per_sample, _ = noise_estim(scen, h_t_noisy)
    if len(h_t_noisy.shape) == 3:
        full_power_per_pilot = (h_t_noisy**2).sum()
    elif len(h_t_noisy.shape) == 4:
        full_power_per_pilot = ((h_t_noisy**2).mean(dim=2)).sum()
    noise_power_est = noise_per_sample*XZ
    signal_power_est = max(full_power_per_pilot-noise_power_est, 1e-10)
    SNR_est = max(torch.log10(signal_power_est/noise_power_est)*10., clamp)
    return SNR_est

def softmax_simple(scen, h_up, ml, ml_version=0, additional_data=None,device=None,dtype=torch.float32):
    assert len(h_up.shape) == 3 and h_up.shape[2] == 2
    power = (h_up[:, :, 0] ** 2 + h_up[:, :, 1] ** 2).sum(dim=0)
    SNR_est = SNR_estim(h_up, scen)
    mean_power = power.mean()
    if ml_version == 0:
        soft_m = torch.exp(ml['softm_main_scale'])
        soft_m = soft_m * torch.softmax((ml['softm_power'] + ml['softm_SNR'] * SNR_est) * power / mean_power, 0)
    elif ml_version == 0.5:
        soft_m = torch.exp(ml['softm_main_scale'])
        soft_m = soft_m * torch.sigmoid((ml['sigm_power'] + ml['sigm_SNR'] * SNR_est) * power / mean_power)
        soft_m = soft_m * torch.softmax((ml['softm_power'] + ml['softm_SNR'] * SNR_est) * power / mean_power, 0)
    elif ml_version == 1:
        soft_m = torch.exp(ml['softm_main_scale'])
        soft_m = soft_m * torch.sigmoid((ml['sigm_power'] + ml['sigm_SNR'] * SNR_est) * power / mean_power)
        soft_m = soft_m * torch.softmax((ml['softm_power'] + ml['softm_SNR'] * SNR_est) * power / mean_power, 0)
        x = torch.arange(len(soft_m), device=device, dtype=dtype)
        soft_m = soft_m * torch.sigmoid(ml['bayes_c0']
                                        + ml['bayes_c1'] * x
                                        )
    elif ml_version == 1:
        soft_m = torch.exp(ml['softm_main_scale'])
        soft_m = soft_m*torch.sigmoid((ml['sigm_power']+ml['sigm_SNR']*SNR_est)*power/mean_power)
        soft_m = soft_m*torch.softmax((ml['softm_power']+ml['softm_SNR']*SNR_est)*power/mean_power, 0)
        if scen.comb==0:
            x = torch.arange(len(soft_m), device=device, dtype=dtype)
            soft_m = soft_m*torch.sigmoid(ml['bayes_c0']+ml['bayes_c1']*x)
        else:
            x = torch.arange(len(soft_m)//2, device=device, dtype=dtype)
            y = torch.zeros(len(soft_m)//2, device=device, dtype=dtype)
            first = torch.cat((x, y), dim=0)
            second = torch.cat((y, x), dim=0)
            soft_m = soft_m*torch.sigmoid(ml['bayes_c0']+ml['bayes_c1']*(first+second))
            
    elif ml_version == 2:
        eps = 1e-2
        h_transformed = 0 * h_up
        h_transformed[:, :, 0] = h_transformed[:, :, 0] + eps * torch.mm(ml['A0'], h_up[:, :, 0]) + eps * torch.mm(
            ml['A1'], h_up[:, :, 1])
        h_transformed[:, :, 1] = h_transformed[:, :, 1] - eps * torch.mm(ml['A1'], h_up[:, :, 0]) + eps * torch.mm(
            ml['A0'], h_up[:, :, 1])
        assert (h_transformed.dtype == dtype)
        power1 = (h_transformed[:, :, 0] ** 2 + h_transformed[:, :, 1] ** 2).sum(dim=0)

        soft_m = torch.exp(ml['softm_main_scale'])
        soft_m = soft_m * torch.sigmoid((ml['sigm_power'] + ml['sigm_SNR'] * SNR_est) * power / mean_power +
                                        (ml['sigm_power1'] + ml['sigm_SNR1'] * SNR_est) * power1 / mean_power)
        soft_m = soft_m * torch.softmax((ml['softm_power'] + ml['softm_SNR'] * SNR_est) * power / mean_power +
                                        (ml['softm_power1'] + ml['softm_SNR1'] * SNR_est) * power1 / mean_power, 0)
        x = torch.arange(len(soft_m), device=device, dtype=dtype)
        soft_m = soft_m * torch.sigmoid(ml['bayes_c0']
                                        + ml['bayes_c1'] * x
                                        )

    elif ml_version == 3:
        soft_m = torch.exp(ml['softm_main_scale'])
        soft_m = soft_m * torch.sigmoid((ml['sigm_power'] + ml['sigm_SNR'] * SNR_est) * power / mean_power)
        soft_m = soft_m * torch.softmax((ml['softm_power'] + ml['softm_SNR'] * SNR_est) * power / mean_power, 0)
        x = torch.arange(len(soft_m), device=device, dtype=dtype)
        soft_m = soft_m * torch.sigmoid(ml['bayes_c0']
                                        + ml['bayes_c1'] * x
                                        )
        dx = len(soft_m) * ml['bayes_pos2'] - x
        dy = len(soft_m) * (ml['bayes_pos2'] - ml['bayes_pos1'])
        factor = (torch.relu(dx) / dy).clamp(0, 1)
        soft_m = soft_m * factor

    elif ml_version == 4:
        soft_m = torch.exp(ml['softm_main_scale'] + ml['coeff_current_iter'] * additional_data['current_iter'])
        soft_m = soft_m * torch.sigmoid((ml['sigm_power'] + ml['sigm_SNR'] * SNR_est) * power / mean_power)
        soft_m = soft_m * torch.softmax((ml['softm_power'] + ml['softm_SNR'] * SNR_est) * power / mean_power, 0)
        x = torch.arange(len(soft_m), device=device, dtype=dtype)
        soft_m = soft_m * torch.sigmoid(ml['bayes_c0']
                                        + ml['bayes_c1'] * x
                                        )

    elif ml_version == 5:
        power_diff = 0 * power
        power_diff[:-1] = power_diff[:-1] + ((h_up[:, 1:, 0] - h_up[:, :-1, 0]) ** 2 +
                                             (h_up[:, 1:, 1] - h_up[:, :-1, 1]) ** 2).sum(dim=0)
        power_diff[-1] = power_diff[-2]

        soft_m = torch.exp(ml['softm_main_scale'])
        soft_m = soft_m * torch.sigmoid(((ml['sigm_power'] + ml['sigm_SNR'] * SNR_est) * power +
                                         (ml['sigm_power_diff'] + ml[
                                             'sigm_SNR_diff'] * SNR_est) * power_diff) / mean_power)
        soft_m = soft_m * torch.softmax(((ml['softm_power'] + ml['softm_SNR'] * SNR_est) * power +
                                         (ml['softm_power_diff'] + ml[
                                             'softm_SNR_diff'] * SNR_est) * power_diff) / mean_power, 0)
        x = torch.arange(len(soft_m), device=device, dtype=dtype)
        soft_m = soft_m * torch.sigmoid(ml['bayes_c0']
                                        + ml['bayes_c1'] * x
                                        )
    return soft_m

def upsampling(scen, h, inverse=False):  # inverse: inverse transform to frequency domain
    N_used = scen.RB_num * scen.RB_size

    if inverse:
        if len(h.shape) == 3:
            h_t_rolled = torch.roll(h, -scen.N_shift, dims=1)
            h_fft = torch.fft(h_t_rolled, signal_ndim=1, normalized=True)
            h_f = torch.roll(h_fft, N_used // 2, dims=1)
        elif len(h.shape) == 4:
            h_f = torch.zeros_like(h)
            for k in range(h.shape[2]):
                h_f[:, :, k] = upsampling(scen, h[:, :, k], inverse=True)
        return h_f

    else:
        XZ = scen.upsample_factor * scen.Nfft
        assert len(h.shape) == 3 or len(h.shape) == 4
        if len(h.shape) == 3:
            assert h.shape[0] == scen.Nrx and h.shape[1] >= N_used and h.shape[-1] == 2
            h_f = torch.zeros((scen.Nrx, XZ, 2))
            h_f[:, :N_used] = h
            h_f_rolled = torch.roll(h_f, -N_used // 2, dims=1)
            h_ifft = torch.ifft(h_f_rolled, signal_ndim=1, normalized=True)
            h_t = torch.roll(h_ifft, scen.N_shift, dims=1)
        elif len(h.shape) == 4:
            assert h.shape[0] == scen.Nrx and h.shape[1] >= N_used and h.shape[-1] == 2
            h_t = torch.zeros((scen.Nrx, XZ, h.shape[2], 2))
            for k in range(h.shape[2]):
                h_t[:, :, k] = upsampling(scen, h[:, :, k], inverse=False)
        return h_t


def extract_peak_simple(scen, h_f, ml, ml_version=0, additional_data=None,device=None,dtype=torch.float32):
    N_used = scen.RB_num*scen.RB_size
    h_t = upsampling(scen, h_f)
    soft_m = softmax_simple(scen, h_t, ml, ml_version=ml_version, additional_data=additional_data,device=device,dtype=dtype)
    h_peak0 = h_t*soft_m.view(1, len(soft_m), 1)
    h_peak_f = upsampling(scen, h_peak0, inverse=True)[:,:N_used]
    return h_peak_f


def vectorized_detector(scen, h_rec_data, h_data, device=None,seed=None):
    UE_power = torch.mean(h_data[:, :, :, 0] ** 2 + h_data[:, :, :, 1] ** 2)

    if seed is not None:
        torch.manual_seed(seed)
    N_pilot_sym = scen.N_pilot*scen.N_TTI
    N_data_sym = (14-N_pilot_sym)*scen.N_TTI
    N_used = scen.RB_num*scen.RB_size
    ## transmision
    s_tx = torch.ones(2, dtype =torch.float32,device=device)/torch.sqrt(torch.tensor([2, 2], dtype = torch.float32,device=device)) #(1+1j)/sqrt(2)
    white_noise_d = torch.randn(h_data.size(), dtype=torch.float32,device=device)
    white_noise_d /= torch.sqrt(torch.tensor(2, dtype=torch.float32,device=device))
    noise_d = torch.sqrt(UE_power) * white_noise_d / torch.sqrt(torch.tensor(10**(scen.SNR/10), dtype=torch.float32,device=device))
    noise_power = torch.mean(noise_d[:, :, :, 0]**2+noise_d[:, :, :, 1]**2)
    data_noisy = torch.zeros(h_data.size(), dtype = torch.float32,device=device)
          
    #receiving
    err_data = torch.zeros(1,dtype=torch.float32,device=device)
    for k in range (N_data_sym):
        data_noisy[:, :, k, 0] = h_data[:, :, k, 0]*s_tx[0] - h_data[:, :, k, 1]*s_tx[1] +  noise_d[:, :, k, 0]
        data_noisy[:, :, k, 1] = h_data[:, :, k, 0]*s_tx[1] + h_data[:, :, k, 1]*s_tx[0] +  noise_d[:, :, k, 1]

        det_data = torch.zeros((N_used,s_tx.size()[0]), dtype = torch.float32,device=device)
        H_re = h_rec_data[:,:, k, 0]
        H_im = -h_rec_data[:,:, k, 1]
        Y = data_noisy[:,:, k, :]
        
        np_plus_power=noise_power+torch.sum(H_re**2+H_im**2,dim=0)
        
        det_data[:,0] = torch.sum(Y[:,:, 0].clone()*H_re - Y[:,:, 1].clone()*H_im,dim=0)/np_plus_power
        det_data[:,1] = torch.sum(Y[:,:, 1].clone()*H_re + Y[:,:, 0].clone()*H_im,dim=0)/np_plus_power
            # раскоментить чтобы проверить корректность работы (должно получаться [0.7 0.7],  как и s_tx)
            # print(det_data.data.numpy())
        err=det_data - s_tx
        err_data=err_data+torch.sum(err**2)

    return err_data


# ## Тест всего алгоритма

def tester(scen, path, ml, max_iter, num_ant,device=None,n_pilots=1,dtype=torch.float32):
    torch.manual_seed(scen.seed*1000)
    scaleFactor = 1e4
    
    N_pilot_sym = scen.N_pilot*scen.N_TTI
    N_data_sym = (14-N_pilot_sym)*scen.N_TTI
    N_used = scen.RB_num*scen.RB_size    
    
    h_pilot, h_data = data_load(scen, path, astype=torch.float32, device=device, n_pilots=n_pilots)

    h_pilot *= scaleFactor
    h_data *= scaleFactor

    UE_power_pilot = 0.5*torch.mean(h_pilot[:, :, 0]**2 + h_pilot[:, :, 1]**2+h_pilot[:, :, 2]**2 + h_pilot[:, :, 3]**2)

    white_noise_p = torch.randn(h_pilot.size(), dtype=torch.float32,device=device)
    white_noise_p /= torch.sqrt(torch.tensor(2, dtype=torch.float32,device=device))
    noise_p = torch.sqrt(UE_power_pilot) * white_noise_p / torch.sqrt(torch.tensor(10**(scen.SNR/10), dtype=torch.float32,device=device))
    h_f_noisy = h_pilot + noise_p

    if n_pilots==2:
        h_f_noisy = (h_f_noisy[:, :, 0:2] + h_f_noisy[:, :, 2:4]) / 2
    
    h_f_recovered_pilots, h_f_recovered_data = CE_TTI(scen, h_f_noisy, ml, max_iter, device=device,dtype=dtype)

    detector_result = vectorized_detector(scen, h_f_recovered_data, h_data, device=device)
    
#     # saving test samples ##############
#     SAVE_MATRICES_PATH = '../tests/detector_matrices/'
#     BASE_FNAME = path.split('/')[-1].split('.')[0]
#     save_args(scen=scen, h_rec_data=h_f_recovered_data, h_data=h_data, UE_power=UE_power_data, result=detector_result, basedir=SAVE_MATRICES_PATH, prefix=BASE_FNAME+'__')
    ####################################
    
    err_data = detector_result/torch.tensor(N_used*N_data_sym,dtype=torch.float32,device=device) 

    return None, err_data


def batches_learning(scen, ml, max_iter, range_SNR,train_files,SNR_error_weights=None,return_SNR_batch_errors=False,
                     device=None,n_pilots=1,dtype=torch.float32):
    scen = scen._replace(N_scenarios=len(train_files))
    
    # default weights of SNR errors
    if SNR_error_weights is None:
        SNR_error_weights = dict([(snr,1.5**i) for i,snr in enumerate(range_SNR)])
    
    N_scenarios=len(train_files)
    N_seeds=scen.N_seeds
    
    SNR_error=torch.tensor(0, dtype=torch.float32,device=device)
    
    SNR_batch_errors = np.zeros(len(range_SNR))
    
    for snr_index,SNR in enumerate(range_SNR):
        s = []        
        err_seed=torch.zeros(N_scenarios*N_seeds, dtype=torch.float32,device=device)  

        for j in range(N_scenarios*N_seeds):
            internal = Scenario(SNR=SNR , seed=random.randint(1,10**6),  RB_num=4, index=np.floor(j/N_seeds)+1,
                    N_TTI=1, UE_indx=0, UE_number=1, N_seeds=N_seeds, N_scenarios=N_scenarios, RB_size=12, N_pilot=scen.N_pilot, Nrx=scen.Nrx,
                    upsample_factor=1, Nfft=512, N_shift=100, N_response=int(448*512/2048), comb=0)
            s.append(internal)
    
        for j,scen_filename in zip(range(N_scenarios*N_seeds),cycle(train_files)):
            sys = s[j]
            _, err_seed[j] = tester(sys, scen_filename, ml, max_iter, 60,device=device,n_pilots=n_pilots,dtype=dtype)#последнее число - номер антенны на которой картинки смотреть
    
        err_scen=torch.sum(err_seed)/torch.tensor(N_seeds,dtype=torch.float32,device=device)
        
        SNR_batch_errors[snr_index] = err_scen.data.detach()
        
        SNR_error=SNR_error+err_scen*torch.tensor(SNR_error_weights[SNR],dtype=torch.float32,device=device)
        
    if return_SNR_batch_errors:
        return SNR_error,SNR_batch_errors
    
    return SNR_error


def test(scen, ml, max_iter, train_files,random_state=None,use_tqdm=False,device=None,n_pilots=1,dtype=torch.float32):
    N_seeds=scen.N_seeds
    SNR = scen.SNR

    all_scen_errors = list()  

    if random_state is not None:
        random.seed(random_state)
    
    if use_tqdm:
        tqdm_func = tqdm
    else:
        tqdm_func = lambda x:x
    
    for j,scen_filename in tqdm_func(enumerate(train_files)):
        scen_errors = list()
        for seed_n in range(N_seeds):
            internal_scen = scen._replace(SNR=SNR)
            internal_scen = internal_scen._replace(seed=random.randint(1,100))
        
            _, err_seed_scen = tester(internal_scen, scen_filename, ml, max_iter, 60, device=device,n_pilots=n_pilots,dtype=dtype)#последнее число - номер антенны на которой картинки смотреть
            scen_errors.append(err_seed_scen.cpu().detach().numpy())
        all_scen_errors.append(np.mean(scen_errors))
            
    return all_scen_errors
