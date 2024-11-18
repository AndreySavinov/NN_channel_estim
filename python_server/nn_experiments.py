from __future__ import print_function
import numpy as np
from scipy.io import loadmat
from collections import namedtuple
import torch
#from torch.nn import functional as F
from matplotlib import pyplot as plt
#from torch.optim import Adam
import os
import hashlib 
#from torchviz import make_dot
import time 
from torch import tensor
import pprint
from scipy.io import loadmat


#------- GLOBAL SETTINGS

#onePilotFolder = '/home/dmitry/current_work/MIMO/hw_ml/exploration/one_pilot_data'
onePilotFolder = 'one_pilot_data'

#deviceType = 'cuda' 
deviceType = 'cpu'

if deviceType == 'cuda':
    print ('Using GPU\n')
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    dtype = torch.float32
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
        
elif deviceType == 'cpu':
    print ('Using CPU\n')
    device = torch.device('cpu')
    dtype = torch.float32# was 64    
    torch.set_default_tensor_type(torch.FloatTensor)#(torch.DoubleTensor)
    

Scenario = namedtuple('Scenario', [
    'SNR', 'seed','index', 'RB_num', 'N_TTI', 'UE_indx', 'UE_number', 
    'N_seeds', 'N_scenarios', 'N_pilot', 'RB_size', 'Nrx', 'upsample_factor', 'Nfft', 'N_shift', 'N_response','comb'])

scen0 = Scenario(SNR=0, seed=0,  RB_num=1, index=1,
                    N_TTI=1, UE_indx=0, UE_number=1, N_seeds=4, N_scenarios=140, RB_size=12, N_pilot=2, Nrx=64,
                    upsample_factor=1, Nfft=512, N_shift=10, N_response=448*512//2048, comb = 1)

# Here base parameters of scenario are defined
# All functions which names start from "demo" or "test" aren't needed for training and loss calculations
# To estimate loss, function 'getLoss' is required
# To start learning, coefficients in 'DemoGradDescent ' should be defined, and just run this script. Parameters of learning are defined   # above and in "__main__" for   'DemoGradDescent '

# Preload all mat files
preload = True

#------- END OF GLOBAL SETTINGS
    
def data_load(scen,
              ind=None, # index of mat file, can be used instead of path
              path=None, 
              scaleFactor=1e4, # rescale data so that it is of order ~1e0  
              use_preloaded=True,
              ):   
    '''
    Returns 4D tensors h_pilot, h_data, with dimensions:   
    1 - antenna number (up to Nrx)
    2 - frequency/time position 
    3 - symbol index (up to N_pilot_sym or N_data_sym)
    4 - real/imag component   
    '''
    comb=scen.comb
    # N-of-pilot-symbols
    N_pilot_sym = scen.N_pilot*scen.N_TTI
    N_data_sym = (14-N_pilot_sym)*scen.N_TTI;
    assert (N_pilot_sym == 1 or N_pilot_sym == 2)
    if N_pilot_sym == 1:
        pilot_positions = [3]
    elif N_pilot_sym == 2:
        pilot_positions = [3, 10]
        
    N_used = scen.RB_num*scen.RB_size    
    
    if path is None:
        assert (not ind is None)
        path = os.path.join(onePilotFolder, 'temp_chan_seed'+str(ind)+'.mat')
        
    if use_preloaded:
        assert (not ind is None)
        return dataL[ind-1]

    H_tmp = loadmat(path)['H_new']
    
    # Extract pilot symbols
    h_pilot = torch.zeros(scen.Nrx, N_used, N_pilot_sym, 2)
    for s in range(N_pilot_sym):
        h_pilot[:,0:N_used:(1+comb),s,0] = torch.sqrt(torch.tensor(1+comb, dtype=dtype))*torch.tensor(H_tmp[0,:scen.Nrx,0:N_used:(1+comb), pilot_positions[s]].real)
        h_pilot[:,0:N_used:(1+comb),s,1] = torch.sqrt(torch.tensor(1+comb, dtype=dtype))*torch.tensor(H_tmp[0,:scen.Nrx,0:N_used:(1+comb), pilot_positions[s]].imag)
    
    # Extract data symbols
    h_data = torch.zeros(scen.Nrx, N_used, N_data_sym, 2)
    s = 0
    for k in range(14):
        if not k in pilot_positions:
            h_data[:,:,s,0] = torch.tensor(H_tmp[0,:scen.Nrx,0:N_used, k].real)
            h_data[:,:,s,1] = torch.tensor(H_tmp[0,:scen.Nrx,0:N_used, k].imag)
            s += 1
    
    h_pilot *= scaleFactor
    h_data *= scaleFactor
    
    return h_pilot, h_data


# Preload all mat files
if preload:
    dataL = []
    for ind in range(1,141):
        path = os.path.join(onePilotFolder, 'temp_chan_seed'+str(ind)+'.mat')
        h_pilot, h_data = data_load(scen=scen0, ind=ind, use_preloaded=False)
        dataL.append([h_pilot, h_data])          
    print ('Preload data: OK')


def demo_data_load():
    #for ind in range(1, 141):
        #print ('-----------', ind)
    h_pilot, h_data = data_load(scen0, ind=1)
    plt.figure(figsize=(10, 7))
    plt.plot(h_pilot[13, :, 0, 0].data.numpy())#14 antenna all subcarriers first pilot, real
    print ('Shapes:', h_pilot.shape, h_data.shape)
    print ('Mean abs. val. squared of pilots:', (h_pilot**2).mean())
    print ('Mean abs. val. squared of data symbols:', (h_data**2).mean())
          
        
def upsampling(scen, h, inverse=False): # inverse: inverse transform to frequency domain
    N_used = scen.RB_num*scen.RB_size
    XZ = scen.upsample_factor*scen.Nfft
    if not inverse:
        assert len(h.shape) == 3 or len(h.shape) == 4
        if len(h.shape) == 3:
            assert h.shape[0] == scen.Nrx and h.shape[1] >= N_used and h.shape[-1] == 2
            h_f = torch.zeros((scen.Nrx, XZ, 2))
            h_f[:,:N_used] = h
            h_f_rolled = torch.roll(h_f, -N_used//2, dims=1)
            h_ifft = torch.ifft(h_f_rolled, signal_ndim=1, normalized=True)
            h_t = torch.roll(h_ifft, scen.N_shift, dims=1)
        elif len(h.shape) == 4:
            assert h.shape[0] == scen.Nrx and h.shape[1] >= N_used and h.shape[-1] == 2
            h_t = torch.zeros((scen.Nrx, XZ, h.shape[2], 2))
            for k in range(h.shape[2]):
                h_t[:,:,k] = upsampling(scen, h[:,:,k], inverse=False)
        return h_t
    
    else:
        if len(h.shape) == 3:
            h_t_rolled = torch.roll(h, -scen.N_shift, dims=1)
            h_fft = torch.fft(h_t_rolled, signal_ndim=1, normalized=True)
            h_f = torch.roll(h_fft, N_used//2, dims=1)
        elif len(h.shape) == 4:
            h_f = torch.zeros_like(h)
            for k in range(h.shape[2]):
                h_f[:,:,k] = upsampling(scen, h[:,:,k], inverse=True)
        return h_f
    
   
def test_upsampling_power_conservation():
    ind = 13
    h_pilot, h_data = data_load(scen0, ind=ind)    
    # all pilots
    h_t = upsampling(scen0, h_pilot) 
    assert torch.allclose((h_pilot**2).sum(), (h_t**2).sum())   
    # first pilot
    h = h_pilot[:,:,0]
    h_t = upsampling(scen0, h) 
    assert torch.allclose((h**2).sum(), (h_t**2).sum())  
    # first data symbol
    h = h_data[:,:,0]
    h_t = upsampling(scen0, h) 
    assert torch.allclose((h**2).sum(), (h_t**2).sum())
        
        
def test_upsampling_inverse():
    ind = 13
    h_pilot, h_data = data_load(scen0, ind=ind)
    # all pilots
    h_t = upsampling(scen0, h_pilot) 
    h_ = upsampling(scen0, h_t, inverse=True) 
    assert torch.allclose((h_pilot**2).sum(), (h_**2).sum())
    assert torch.allclose(h_pilot, h_[:,:h_pilot.shape[1]])  
    # first pilot
    h = h_pilot[:,:,0]
    h_t = upsampling(scen0, h) 
    h_ = upsampling(scen0, h_t, inverse=True) 
    assert torch.allclose((h**2).sum(), (h_**2).sum())
    assert torch.allclose(h, h_[:,:h_pilot.shape[1]]) 
    # all data symbols
    h_t = upsampling(scen0, h_data) 
    h_ = upsampling(scen0, h_t, inverse=True) 
    assert torch.allclose((h_data**2).sum(), (h_**2).sum())
    assert torch.allclose(h_data, h_[:,:h_data.shape[1]]) 


def demo_upsampling():
    ind = 13
    N_used = scen0.RB_num*scen0.RB_size
    h_pilot, h_data = data_load(scen0, ind=ind)
    h_t = upsampling(scen0, h_pilot)
    h_f = upsampling(scen0, h_t, inverse=True)
    comb=scen0.comb
    print ('dtype:', h_t.dtype)
    assert (h_t.dtype == dtype)
    print ('Shape:', h_t.shape)
    print ('Mean abs. val. squared, original:', 2*(h_pilot**2).mean())
    print ('Mean abs. val. squared, time domain:', 2*(h_t**2).mean())
    print ('Mean abs. val. squared, freq domain :', 2*(h_f[:, :N_used, :, :]**2).mean())
    print ('rel Error', (h_f[:, 0:N_used:(1+comb), :]**2-(h_pilot[:, 0:N_used:(1+comb), :]**2)).sum()/((h_pilot[:, 0:N_used:(1+comb), :]**2).sum()))
    plt.figure(figsize = (10, 7))
    plt.subplot(311)
    plt.title('Mean power in frequency domain')
    plt.plot(2*(h_pilot**2).mean(dim=(0,2,3)).numpy())
    plt.subplot(312)
    plt.title('Mean power in time domain')
    plt.plot(2*(h_t**2).mean(dim=(0,2,3)).numpy())
    plt.subplot(313)
    plt.title('Mean power in frequency domain transformed')
    plt.plot(2*(h_f[:, :N_used, :, :]**2).mean(dim=(0,2,3)).numpy()) 
    plt.show()  

def getSeed(h, seed):
    '''Form a new seed, given another seed and h'''
    t = (str(float(h.sum()))+'_'+str(h.shape[2])+'_'+str(seed)).encode()
    hash_ = hashlib.md5(t).hexdigest() # get hash
    as_int = int(hash_, 16)
    short = as_int % (1 << 16)
    return short


def testGetSeed():
    ind = 10
    h_pilot, _ = data_load(scen0, ind=ind)
    print (getSeed(h_pilot, 0))
    assert (getSeed(h_pilot, 0) != getSeed(h_pilot, 1))
    seed0 = 0
    seed1 = 0 
    h_pilot_copy = 1*h_pilot
    assert (getSeed(h_pilot_copy, seed1) == getSeed(h_pilot, seed0))

    
def add_noise(h, SNR, seed=None, scen=scen0):#only for pilot symbols
    N_used = scen.RB_num*scen.RB_size
    if not seed is None:
        torch.manual_seed(getSeed(h, seed))
    UE_power = 2*(h**2).mean() # 2 because of averaging over real and imag components
    white_noise_p = torch.randn(h.size())
    white_noise_p /= np.sqrt(2)
    noise_p = torch.sqrt(UE_power) * white_noise_p / np.sqrt(10**(SNR/10.))
    h_noisy = h + noise_p
    if scen.comb == 1:
        comb_matrix=torch.ones(scen.Nrx, N_used, scen.N_pilot,2, dtype=dtype)
        comb_matrix[:,1:N_used:2,:, :]=torch.zeros(scen.Nrx, N_used//2, scen.N_pilot, 2, dtype=dtype)
        h_noisy=h_noisy*comb_matrix
    assert h_noisy.dtype == dtype
    noise_power = 2*(noise_p**2).mean()
    return h_noisy, noise_power  

def add_noise_data(h, SNR, seed=None):#only for data symbols
    if not seed is None:
        torch.manual_seed(getSeed(h, seed))
    UE_power = 2*(h**2).mean() # 2 because of averaging over real and imag components
    white_noise_p = torch.randn(h.size())
    white_noise_p /= np.sqrt(2)
    noise_p = torch.sqrt(UE_power) * white_noise_p / np.sqrt(10**(SNR/10.))
    h_noisy = h + noise_p
    assert h_noisy.dtype == dtype
    noise_power = 2*(noise_p**2).mean()
    return h_noisy, noise_power

def add_noise_matlab(h, SNR, ind, seed=None, scen=scen0):#only for pilot symbols
    N_used = scen.RB_num*scen.RB_size
    if not seed is None:
        torch.manual_seed(getSeed(h, seed))
    UE_power = 2*(h**2).mean() # 2 because of averaging over real and imag components
    path = 'matlab_noise/noise_'+str(ind)+'.mat'
    noise_p = loadmat(path)['noise_p']
    h_noisy = torch.zeros_like(h)
    h_noisy[:,:,:,0] = h[:, :, :, 0] + torch.tensor(noise_p.real, dtype=dtype)
    h_noisy[:,:,:,1] = h[:, :, :, 1] + torch.tensor(noise_p.imag, dtype=dtype)
    if scen.comb == 1:
        comb_matrix=torch.ones(scen.Nrx, N_used, scen.N_pilot,2, dtype=dtype)
        comb_matrix[:,1:N_used:2,:, :]=torch.zeros(scen.Nrx, N_used//2, scen.N_pilot, 2, dtype=dtype)
        h_noisy=h_noisy*comb_matrix
    assert h_noisy.dtype == dtype
    noise_power = 2*(noise_p**2).mean()
    return h_noisy, noise_power  


def demo_add_noise(ind):
    h_pilot, h_data = data_load(scen0, ind=ind)
    h_noisy, _, noise_p= add_noise(h_pilot, -12.) 
    #print ('Mean power original:', 2*(h_pilot**2).mean()) 
    #print ('Mean power noisy:', 2*(h_noisy**2).mean()) 
    print ('Mean power noisy:', (noise_p**2).sum().data.numpy()) #


def noise_estim(scen, h_t_noisy):
    XZ = scen.upsample_factor*scen.Nfft
    length = int(XZ*((1./2-1./16) - (1./4+1./16)))+int(XZ*((1.-1./16) - (3./4+1./16)))
    x = int(XZ*((1./2-1./16) - (1./4+1./16)))
    y = int(XZ*(1./4+1./16))
    w = int(XZ*(3./4+1./16))
    z = int(XZ*(1.-1./16))
     
    assert (len(h_t_noisy.shape) == 3 or len(h_t_noisy.shape) == 4)
    if len(h_t_noisy.shape) == 3:
        noise_array = torch.zeros((scen.Nrx, length, 2))
        noise_array[:, :x, :] = h_t_noisy[:,y:x+y]
        noise_array[:, x:, :] = h_t_noisy[:,w:z]        
        noise_tmp = noise_array[:, :, 0]**2+noise_array[:, :, 1]**2
    elif len(h_t_noisy.shape) == 4:       
        noise_tmp = torch.zeros((scen.Nrx, length))
        noise_tmp[:, :x] = 2*((h_t_noisy**2)[:,y:x+y]).mean(dim=(2,3))
        noise_tmp[:, x:] = 2*((h_t_noisy**2)[:,w:z]).mean(dim=(2,3))
      
    noise_per_sc=torch.sum(noise_tmp)/(scen.N_pilot*4.)#noise per subcarrier
    NS2 = noise_per_sc/XZ
    noise_per_sample1= torch.sum(torch.sum(noise_tmp[0:scen.Nrx:2,:],dim=0))/(XZ/4.)
    noise_per_sample2= torch.sum(torch.sum(noise_tmp[1:scen.Nrx:2,:],dim=0))/(XZ/4.)
    noise_per_sample = noise_per_sample1+noise_per_sample2
    return noise_per_sample, NS2
    

def demo_noise_estim():
    ind = 13
    h_pilot, _ = data_load(scen0, ind=ind)
    h_noisy, _ = add_noise(h_pilot, SNR=0) 
    h_t_noisy = upsampling(scen0, h_noisy)
    plt.plot((h_noisy**2).mean(dim=(0,2,3)).numpy())
    plt.show()
    noise_per_sample, NS2 = noise_estim(scen0, h_t_noisy)
    print ('Mean power original:', 2*(h_pilot**2).mean()) 
    print ('Mean power noisy:', 2*(h_noisy**2).mean())
    print ('Total power noisy:', (h_noisy**2).sum())
    print ('Mean power noisy time domain:', 2*(h_t_noisy**2).mean())
    print ('Total power noisy time domain:', (h_t_noisy**2).sum())
    print ('noise_per_sample, NS2:', noise_per_sample, NS2) 
    print ('noise_per_sample / NS2:', noise_per_sample/NS2)
    print ('noise_per_sample / Mean power noisy:', noise_per_sample/(2*h_t_noisy**2).mean())
  

def demo_SNR_noise():
    ''' Check dependence  SNR vs NS2/mean_power found using noise_estim()'''
    SNR_L = []
    log_NS2_to_mean_power_L = []
    for ind in range(1, 141):
        print ('------', ind)
        h_pilot, _ = data_load(scen0, ind=ind)
        SNR = int(-30+60*np.random.rand())
        print ('SNR:', SNR)
        SNR_L.append(SNR)
        h_noisy,_ = add_noise(h_pilot, SNR) 
        h_t_noisy = upsampling(scen0, h_noisy)
        mean_power = 2*(h_t_noisy**2).mean()
        _, NS2 = noise_estim(scen0, h_t_noisy)  
        log_NS2_to_mean_power = float(np.log(NS2/mean_power)) 
        log_NS2_to_mean_power_L.append(log_NS2_to_mean_power)
        print ('log(NS2/mean_power):', log_NS2_to_mean_power)
        
    plt.title('SNR vs noise estimate for 140 matfiles (SNRs are chosen randomly)')
    plt.plot(SNR_L[:40], log_NS2_to_mean_power_L[:40], 's', label='1-40')
    plt.plot(SNR_L[40:80], log_NS2_to_mean_power_L[40:80], 'o', label='41-80')
    plt.plot(SNR_L[80:140], log_NS2_to_mean_power_L[80:140], '*', label='81-140')
    plt.xlabel('SNR')
    plt.ylabel('log(NS2/mean_power)')
    plt.legend()
    plt.show()
     

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
    
    
def test_SNR_estim():
    SNR_true_L = []
    SNR_est_L = []
    for ind in range(1, 141):
        print ('------', ind)
        h_pilot, _ = data_load(scen0, ind=ind)
        SNR = int(-30+60*np.random.rand())
        print ('SNR:', SNR)
        SNR_true_L.append(SNR)
        h_noisy,_ = add_noise(h_pilot, SNR) 
        h_t_noisy = upsampling(scen0, h_noisy)
        SNR_est = SNR_estim(h_t_noisy, scen0)
        print ('SNR estimated:', SNR_est)
        SNR_est_L.append(SNR_est)

        
    plt.title('SNR true vs estimated using noise_estim from 140 matfiles (SNRs are chosen randomly)')
    plt.plot(SNR_true_L[:40], SNR_est_L[:40], 's', label='1-40')
    plt.plot(SNR_true_L[40:80], SNR_est_L[40:80], 'o', label='41-80')
    plt.plot(SNR_true_L[80:140], SNR_est_L[80:140], '*', label='81-140')
    plt.plot([-30, 30], [-30,30],'-', label='x=y')
    plt.xlabel('SNR true')
    plt.ylabel('SNR estimated')
    plt.legend()
    plt.show()    


def softmax_simple(scen, h_up, ml, ml_version=0, additional_data=None):
    assert len(h_up.shape) == 3 and h_up.shape[2] == 2
    SNR_est = SNR_estim(h_up, scen)
    XZ = scen.upsample_factor*scen.Nfft
    power = (h_up[:, :, 0]**2+h_up[:, :, 1]**2).sum(dim=0)
    mean_power = power.mean()
    if ml_version == 0:
        soft_m = torch.exp(ml['softm_main_scale'])
        soft_m = soft_m*torch.softmax((ml['softm_power']+ml['softm_SNR']*SNR_est)*power/mean_power, 0)
    elif ml_version == 0.5:
        soft_m = torch.exp(ml['softm_main_scale'])
        soft_m = soft_m*torch.sigmoid((ml['sigm_power']+ml['sigm_SNR']*SNR_est)*power/mean_power)
        soft_m = soft_m*torch.softmax((ml['softm_power']+ml['softm_SNR']*SNR_est)*power/mean_power, 0)
    elif ml_version == 0.7:        
        soft_m = torch.exp(ml['softm_main_scale'])
        soft_m = soft_m*torch.sigmoid((ml['sigm_power']+ml['sigm_SNR']*SNR_est+ml['sigm_SNR1']*SNR_est**2)*power/mean_power)
        soft_m = soft_m*torch.softmax((ml['softm_power']+ml['softm_SNR']*SNR_est+ml['softm_SNR1']*SNR_est**2)*power/mean_power, 0)
    elif ml_version == 0.8:   
        common_power = (h_up[:, :, :].sum(dim=0)**2).sum(dim=-1)     
        soft_m = torch.exp(ml['softm_main_scale'])
        soft_m = soft_m*torch.sigmoid((ml['sigm_power']+ml['sigm_SNR']*SNR_est+ml['sigm_SNR1']*SNR_est**2)*power/mean_power+
                                      (ml['common_power_sigm']+ml['common_power_sigm_SNR']*SNR_est)*common_power/mean_power)
        soft_m = soft_m*torch.softmax((ml['softm_power']+ml['softm_SNR']*SNR_est+ml['softm_SNR1']*SNR_est**2)*power/mean_power+
                                      (ml['common_power_softm']+ml['common_power_softm_SNR']*SNR_est)*common_power/mean_power, 0)
    elif ml_version == 1:
        soft_m = torch.exp(ml['softm_main_scale'])
        soft_m = soft_m*torch.sigmoid((ml['sigm_power']+ml['sigm_SNR']*SNR_est)*power/mean_power)
        soft_m = soft_m*torch.softmax((ml['softm_power']+ml['softm_SNR']*SNR_est)*power/mean_power, 0)
        x = torch.arange(len(soft_m), device=device, dtype=dtype)
        soft_m = soft_m*torch.sigmoid(ml['bayes_c0']
                                      +ml['bayes_c1']*x
                                      )            
    elif ml_version == 2: 
        eps = 1e-2
        h_transformed = 0*h_up 
        h_transformed[:,:,0] = h_transformed[:,:,0]+eps*torch.mm(ml['A0'], h_up[:,:,0])+eps*torch.mm(ml['A1'], h_up[:,:,1])
        h_transformed[:,:,1] = h_transformed[:,:,1]-eps*torch.mm(ml['A1'], h_up[:,:,0])+eps*torch.mm(ml['A0'], h_up[:,:,1])
        assert (h_transformed.dtype == dtype) 
        power1 = (h_transformed[:, :, 0]**2+h_transformed[:, :, 1]**2).sum(dim=0)
        
        soft_m = torch.exp(ml['softm_main_scale'])
        soft_m = soft_m*torch.sigmoid((ml['sigm_power']+ml['sigm_SNR']*SNR_est)*power/mean_power+
                                      (ml['sigm_power1']+ml['sigm_SNR1']*SNR_est)*power1/mean_power)
        soft_m = soft_m*torch.softmax((ml['softm_power']+ml['softm_SNR']*SNR_est)*power/mean_power+
                                      (ml['softm_power1']+ml['softm_SNR1']*SNR_est)*power1/mean_power, 0)
        if scen.comb==0:
            x = torch.arange(len(soft_m), device=device, dtype=dtype)
        else:
            z = torch.arange(len(soft_m)//2, device=device, dtype=dtype)
            y = torch.zeros(len(soft_m)//2, device=device, dtype=dtype)
            first = torch.cat((z, y), dim=0)
            second = torch.cat((y, z), dim=0)
            x = first+second
        soft_m = soft_m*torch.sigmoid(ml['bayes_c0']
                                      +ml['bayes_c1']*x
                                      ) 
        
    elif ml_version == 3:         
        soft_m = torch.exp(ml['softm_main_scale'])
        soft_m = soft_m*torch.sigmoid((ml['sigm_power']+ml['sigm_SNR']*SNR_est)*power/mean_power)
        soft_m = soft_m*torch.softmax((ml['softm_power']+ml['softm_SNR']*SNR_est)*power/mean_power, 0)
        if scen.comb==0:
            x = torch.arange(len(soft_m), device=device, dtype=dtype)
        else:
            z = torch.arange(len(soft_m)//2, device=device, dtype=dtype)
            y = torch.zeros(len(soft_m)//2, device=device, dtype=dtype)
            first = torch.cat((z, y), dim=0)
            second = torch.cat((y, z), dim=0)
            x = first+second
        soft_m = soft_m*torch.sigmoid(ml['bayes_c0']
                                      +ml['bayes_c1']*x
                                      )  
        dx = len(soft_m)*ml['bayes_pos2']-x 
        dy = len(soft_m)*(ml['bayes_pos2']-ml['bayes_pos1'])
        factor = (torch.relu(dx)/dy).clamp(0,1)        
        soft_m = soft_m*factor
        
    elif ml_version == 4:
        soft_m = torch.exp(ml['softm_main_scale']+ml['coeff_current_iter']*additional_data['current_iter'])
        soft_m = soft_m*torch.sigmoid((ml['sigm_power']+ml['sigm_SNR']*SNR_est)*power/mean_power)
        soft_m = soft_m*torch.softmax((ml['softm_power']+ml['softm_SNR']*SNR_est)*power/mean_power, 0)
        if scen.comb==0:
            x = torch.arange(len(soft_m), device=device, dtype=dtype)
        else:
            z = torch.arange(len(soft_m)//2, device=device, dtype=dtype)
            y = torch.zeros(len(soft_m)//2, device=device, dtype=dtype)
            first = torch.cat((z, y), dim=0)
            second = torch.cat((y, z), dim=0)
            x = first+second
        soft_m = soft_m*torch.sigmoid(ml['bayes_c0']
                                      +ml['bayes_c1']*x
                                      ) 
        
    elif ml_version == 5:
        power_diff = 0*power 
        power_diff[:-1] = power_diff[:-1] +((h_up[:, 1:, 0]-h_up[:, :-1, 0])**2+
                                                (h_up[:, 1:, 1]-h_up[:, :-1, 1])**2).sum(dim=0)
        power_diff[-1] = power_diff[-2]
                                                
        soft_m = torch.exp(ml['softm_main_scale'])
        soft_m = soft_m*torch.sigmoid(((ml['sigm_power']+ml['sigm_SNR']*SNR_est)*power+
                                       (ml['sigm_power_diff']+ml['sigm_SNR_diff']*SNR_est)*power_diff)/mean_power)
        soft_m = soft_m*torch.softmax(((ml['softm_power']+ml['softm_SNR']*SNR_est)*power+
                                       (ml['softm_power_diff']+ml['softm_SNR_diff']*SNR_est)*power_diff)/mean_power, 0)
        if scen.comb==0:
            x = torch.arange(len(soft_m), device=device, dtype=dtype)
        else:
            z = torch.arange(len(soft_m)//2, device=device, dtype=dtype)
            y = torch.zeros(len(soft_m)//2, device=device, dtype=dtype)
            first = torch.cat((z, y), dim=0)
            second = torch.cat((y, z), dim=0)
            x = first+second
        soft_m = soft_m*torch.sigmoid(ml['bayes_c0']
                                      +ml['bayes_c1']*x
                                      )  
    elif ml_version == 6:   
        max_power = torch.max(power)     
        soft_m = torch.exp(ml['softm_main_scale']+ml['softm_SNR_scale']*SNR_est+ml['max_power_scale']*max_power/mean_power)
        soft_m = soft_m*torch.sigmoid((ml['sigm_power']+ml['max_power_sigm']*max_power/mean_power+
                                       ml['sigm_SNR']*SNR_est+ml['sigm_SNR1']*SNR_est**2)*power/mean_power)
        soft_m = soft_m*torch.softmax((ml['softm_power']+ml['max_power_softm']*max_power/mean_power+
                                       ml['softm_SNR']*SNR_est+ml['softm_SNR1']*SNR_est**2)*power/mean_power, 0) 
        
    elif ml_version in [8, 12]:   
        max_power = torch.max(power)
        if scen.comb==0:
            x = torch.arange(len(power), device=device, dtype=dtype)
        else:
            z = torch.arange(len(power)//2, device=device, dtype=dtype)
            y = torch.zeros(len(power)//2, device=device, dtype=dtype)
            first = torch.cat((z, y), dim=0)
            second = torch.cat((y, z), dim=0)
            x = first+second
            
        soft_m = torch.exp(ml['softm_main_scale']+ml['softm_SNR_scale']*SNR_est+ml['max_power_scale']*max_power/mean_power
                          +ml['bayes_exp_c0']+ml['bayes_exp_c1']*x)
        soft_m = soft_m*torch.sigmoid((ml['sigm_power']+ml['max_power_sigm']*max_power/mean_power+
                                       ml['sigm_SNR']*SNR_est+ml['sigm_SNR1']*SNR_est**2)*power/mean_power
                                      +ml['bayes_sigm_c0']+ml['bayes_sigm_c1']*x)
        soft_m = soft_m*torch.softmax((ml['softm_power']+ml['max_power_softm']*max_power/mean_power+
                                       ml['softm_SNR']*SNR_est+ml['softm_SNR1']*SNR_est**2)*power/mean_power
                                      +ml['bayes_softm_c0']+ml['bayes_softm_c1']*x, 0)
        
        #soft_m = torch.exp(ml[15]+ml[14]*SNR_est+ml[6]*max_power/mean_power
        #                   +ml[0]+ml[1]*x)
        #soft_m = soft_m*torch.sigmoid((ml[11]+ml[7]*max_power/mean_power+
        #                               ml[9]*SNR_est+ml[10]*SNR_est**2)*power/mean_power
        #                               +ml[2]+ml[3]*x)
        #soft_m = soft_m*torch.softmax((ml[16]+ml[8]*max_power/mean_power+
        #                               ml[12]*SNR_est+ml[13]*SNR_est**2)*power/mean_power
        #                               +ml[4]+ml[5]*x, 0)
        
    elif ml_version == 9:   
        max_power = torch.max(power) 
        power_diff = 0*power 
        power_diff[:-1] = power_diff[:-1] +((h_up[:, 1:, 0]-h_up[:, :-1, 0])**2+
                                                (h_up[:, 1:, 1]-h_up[:, :-1, 1])**2).sum(dim=0)
        power_diff[-1] = power_diff[-2]
        max_p_ant = (h_up[:, :, 0]**2+h_up[:, :, 1]**2).max(dim=0)[0]
            
        if scen.comb==0:
            x = torch.arange(len(power), device=device, dtype=dtype)
        else:
            z = torch.arange(len(power)//2, device=device, dtype=dtype)
            y = torch.zeros(len(power)//2, device=device, dtype=dtype)
            first = torch.cat((z, y), dim=0)
            second = torch.cat((y, z), dim=0)
            x = first+second
        soft_m = torch.exp(ml['softm_main_scale']+ml['softm_SNR_scale']*SNR_est
                           +ml['max_power_scale']*max_power/mean_power
                           +ml['power_diff_exp']*power_diff/mean_power
                           +ml['max_p_ant_exp']*max_p_ant/mean_power
                           +ml['bayes_exp_c0']+ml['bayes_exp_c1']*x)
        soft_m = soft_m*torch.sigmoid((ml['sigm_power']+ml['max_power_sigm']*max_power/mean_power+
                                       ml['sigm_SNR']*SNR_est+ml['sigm_SNR1']*SNR_est**2)*power/mean_power
                                       +ml['power_diff_sigm']*power_diff/mean_power
                                       +ml['max_p_ant_sigm']*max_p_ant/mean_power
                                       +ml['bayes_sigm_c0']+ml['bayes_sigm_c1']*x)
        soft_m = soft_m*torch.softmax((ml['softm_power']+ml['max_power_softm']*max_power/mean_power+
                                       ml['softm_SNR']*SNR_est+ml['softm_SNR1']*SNR_est**2)*power/mean_power
                                       +ml['power_diff_softm']*power_diff/mean_power
                                       +ml['max_p_ant_softm']*max_p_ant/mean_power
                                       +ml['bayes_softm_c0']+ml['bayes_softm_c1']*x, 0) 
                
       
    return soft_m
    

    
    
def extract_peak_simple(scen, h_f, ml, ml_version=0, additional_data=None):    
    N_used = scen.RB_num*scen.RB_size
    XZ = scen.upsample_factor*scen.Nfft
    h_t = upsampling(scen, h_f)
    #plt.figure(figsize=(20, 5))#
    #plt.plot(h_t[13, :, 0].data.numpy(), label='intial')#
    soft_m = softmax_simple(scen, h_t, ml, ml_version=ml_version, additional_data=additional_data)
    h_peak0 = h_t*soft_m.view(1, len(soft_m), 1)
    
    #plt.plot(h_peak0[13, :, 0].data.numpy(), label='softmax')#
    #plt.legend()#
    h_peak_f_minus = upsampling(scen, h_peak0, inverse=True)[:,:N_used]
    if scen.comb==1:
        h_peak_f_plus1 = h_peak_f_minus
        h_peak_f_plus2 = upsampling(scen, torch.cat((h_peak0[:, :XZ//2],-h_peak0[:, XZ//2:]), dim=1), inverse=True)[:,:N_used]
        h_peak_f_plus = (h_peak_f_plus1+h_peak_f_plus2)/torch.sqrt(torch.tensor(2.))
    else:
        h_peak_f_plus = h_peak_f_minus
    
    return h_peak_f_minus, h_peak_f_plus

    
def recover_h(scen, h_f, ml, max_iter, ml_version=1):
    if len(h_f.shape) == 4:
        h_f = h_f.mean(dim=2)
        
    if ml_version == 'trivial':
        return 0*h_f
                
    if ml_version == 'ref':
        h_rec = torch.mean(h_f, dim=1, keepdim=True)*torch.ones((1, h_f.shape[1], 1))
        assert h_rec.shape == h_f.shape
        assert torch.allclose(h_rec.mean(dim=1), h_f.mean(dim=1))
        return h_rec
    
    if 'Nrepeats_' in ml.keys():
        extracted_peaks = [0 * h_f for _ in range(max_iter)]
        recovered_peaks = [0 * h_f for _ in range(max_iter)]
        for _ in range(int(ml['Nrepeats_'])):
            for it in range(max_iter):
                additional_data = {'current_iter': it}
                h_f = h_f + extracted_peaks[it]
                h_recovered_current_minus, h_recovered_current_plus = extract_peak_simple(scen, h_f, ml, ml_version=ml_version,
                                                                      additional_data=additional_data)
                h_f = h_f - h_recovered_current_minus
                extracted_peaks[it] = h_recovered_current_minus
                recovered_peaks[it] = h_recovered_current_plus

        h_f_recovered = 0 * h_f
        for it in range(max_iter):
            h_f_recovered = h_f_recovered + recovered_peaks[it]
        
        return h_f_recovered
            
            
    h_f_recovered = 0*h_f
    for it in range(max_iter): 
        additional_data = {'current_iter': it}       
        h_recovered_current_minus, h_recovered_current_plus = extract_peak_simple(scen, h_f, ml, ml_version=ml_version, 
                                                                                  additional_data=additional_data)            
        h_f = h_f - h_recovered_current_minus
        h_f_recovered = h_f_recovered + h_recovered_current_plus
        
    return h_f_recovered    
    
    
def test_recover_h(): # TODO: update for 2 pilots
    N_used = scen0.RB_num*scen0.RB_size
    ind = 13
    h_pilot, _ = data_load(scen0, os.path.join(onePilotFolder, 'temp_chan_seed'+str(ind)+'.mat'))
    h_noisy_f = add_noise(h_pilot, SNR=-0, seed=0) 
    h_noisy_t = upsampling(scen0, h_noisy_f) 
    h_remaining_t = h_noisy_t
    
    ml = {'softm_main_scale': torch.tensor(2.1, requires_grad=True), 
          'softm_power': torch.tensor(1.5, requires_grad=True),
          'softm_SNR': torch.tensor(-0.3, requires_grad=True),
          'sigm_power': torch.tensor(0., requires_grad=True),
          'sigm_SNR': torch.tensor(-0.3, requires_grad=True),
          'bayes_c0': torch.tensor(0.04, requires_grad=True),
          'bayes_c1': torch.tensor(0.7, requires_grad=True),
          }
    ml_version = 1
    
    print ('Total power:', float((h_noisy_f**2).sum()))
    print ('Signal power:', float((h_pilot**2).sum()))
    print ('Noise power:', float((h_noisy_f**2-h_pilot**2).sum()))
    for max_iter in range(1,5):
        print ('======= Iteration', max_iter)        
        h_recovered = recover_h(scen0, h_noisy_f, ml, max_iter=max_iter, ml_version=ml_version) 
        print ('Recovery error:', float(((h_pilot-h_recovered)**2).sum())/float((h_pilot**2).sum()))
        
        plt.figure(figsize=(12,10))
        
             
        plt.subplot(211)
        plt.title('Iteration '+str(max_iter))   
        for ant in range(3):
            plt.plot(h_pilot[ant,:,0].numpy(), ['r','b','g'][ant]+'--', label='ant '+str(ant)+' original')
            plt.plot(h_noisy_f[ant,:,0].numpy(), ['r','b','g'][ant]+'-', label='ant '+str(ant)+' noisy')
            plt.plot(h_recovered[ant,:N_used,0].detach().numpy(), 
                     ['r','b','g'][ant]+':', label='ant '+str(ant)+' extracted')
        plt.legend(loc='lower right')
        plt.xlabel('Frequency domain, first 3 antennas') 
        
        h_recovered_t = upsampling(scen0, h_recovered)       
        plt.subplot(212)  
        plt.plot((h_remaining_t[:,:,0]**2+h_remaining_t[:,:,1]**2).mean(dim=0).detach().numpy(), '-', label='total before')        
        h_remaining_t = h_noisy_t-h_recovered_t
        plt.plot((h_remaining_t[:,:,0]**2+h_remaining_t[:,:,1]**2).mean(dim=0).detach().numpy(), '-', label='total after')
        plt.plot((h_recovered_t[:,:,0]**2+h_recovered_t[:,:,1]**2).mean(dim=0).detach().numpy(), '-', label='recovered')
        plt.xlabel('Time domain, total power before/aftrer extraction')
        plt.legend()
        plt.savefig(str(max_iter)+'.png')
    plt.show()  

def getLegendre(l, deg):
    Leg_ = torch.zeros((l, deg))
    x_ = torch.linspace(-1,1,l)
    Leg_[:,0] = 1.
    Leg_[:,1] = x_
    for d in range(2, deg):
        Leg_[:,d] = ((2.*d-1)*x_*Leg_[:,d-1]-(d-1)*Leg_[:,d-2])/(d+0.) # Legendre polynomials
    Leg, _ = torch.qr(Leg_) # improve orthogonality 
    return Leg[:,:deg]
    

def poly_projection(scen, h_f, ml, ml_version='poly0', debug=False):
    if ml_version == 'poly0':
        total_power = (h_f**2).sum()
        h_rescaled = h_f/torch.sqrt(total_power)
#         Leg_ = torch.zeros((h_f.shape[1], ml['deg_']))
#         x_ = torch.linspace(-1,1,h_f.shape[1])
#         Leg_[:,0] = 1.
#         Leg_[:,1] = x_
#         for d in range(2, ml['deg_']):
#             Leg_[:,d] = ((2.*d-1)*x_*Leg_[:,d-1]-(d-1)*Leg_[:,d-2])/(d+0.) # Legendre polynomials
#         Leg, _ = torch.qr(Leg_) # improve orthogonality 
        if debug:
            for d in range(0,ml['deg_'],3):
                plt.plot(Leg[:,d].cpu().numpy(), label=str(d))
            plt.legend()
            plt.show()  
        h_recovered_coeffs = torch.tensordot(h_rescaled, Leg[:,:ml['deg_']], dims=([1],[0]))
        assert (h_recovered_coeffs.shape == (h_f.shape[0], 2, ml['deg_']))

        coeff_powers = (h_recovered_coeffs**2).sum(dim=(0,1))
        rescale_coeffs = torch.exp(ml['c0']+ml['c1']*coeff_powers)
        h_recovered_coeffs = h_recovered_coeffs*rescale_coeffs
        
        h_recovered = torch.tensordot(h_recovered_coeffs, Leg[:,:ml['deg_']], dims=([2],[1]))
        h_recovered = h_recovered.permute(0, 2, 1)
        assert (h_recovered.shape == h_f.shape)
        h_recovered = h_recovered * torch.sqrt(total_power)
        #print ((h_f**2).sum(), (h_recovered**2).sum())
        return h_recovered            
    
    else:
        raise NotImplementedError
    
def test_poly_projection():
    h_f = torch.ones((4,40,2))
    ml = {'deg_': 3}
    poly_projection(0, h_f, ml, ml_version='poly0')    
    
def getLoss(ml, 
            lossVersion=None, # 'detector' or 'relError'
            inds=None, 
            SNR_L=None, 
            seed=None, 
            max_iter=5, 
            ml_version=None, 
            SNRscaleFactor=1.,
            scen=scen0, scale=True):
    assert lossVersion in ['detector', 'relError']
    N_used = scen.RB_num*scen.RB_size
    loss = 0
    comb = scen.comb
    if lossVersion == 'relError':
        for ind in inds:
            h_pilot, h_data = data_load(scen, ind=ind)  
            for SNR in SNR_L:           
                h_noisy,_ = add_noise(h_pilot, SNR, seed=seed) 
                h_recovered_ = recover_h(scen, h_noisy, ml, max_iter=max_iter, ml_version=ml_version)                
                
                h_recovered = 0*h_pilot
                for k in range(h_pilot.shape[2]):
                    h_recovered[:,:,k] = h_recovered_
          
                loss_current = ((h_recovered[:, :, 0, :]-h_data[:, :, 0, :])**2).sum()/(h_data[:, :, 0, :]**2).sum()
                loss = loss+SNRscaleFactor**SNR*loss_current
                
    elif lossVersion == 'detector':
        N_pilot_sym = scen.N_pilot*scen.N_TTI
        N_data_sym = (14-N_pilot_sym)*scen.N_TTI;        
        for ind in inds:
            h_pilot, h_data = data_load(scen, ind=ind)         
            for SNR in SNR_L:           
                h_pilot_noisy, _ = add_noise(h_pilot, SNR, seed=seed) 
                h_pilot_rec = recover_h(scen, h_pilot_noisy, ml, 
                                        max_iter=max_iter, ml_version=ml_version)
    
                h_data_noisy, data_noise_power = add_noise_data(h_data, SNR, seed=seed) 
                               
                assert h_pilot_rec.shape[1] == N_used
                H_re = h_pilot_rec[:, :, 0]
                H_im = -h_pilot_rec[:, :, 1]
                
              
                err_data = 0
                for k in range (N_data_sym):    
                    det_data = torch.zeros((N_used, 2))   
                    assert h_data_noisy.shape[1] == N_used 
                    Y = h_data_noisy[:, :, k, :]                    
                    det_data[:,0] = (torch.sum(Y[:,:,0]*H_re-Y[:,:,1]*H_im, dim=0)/
                                    (data_noise_power+torch.sum(H_re**2+H_im**2, dim=0)))
                    det_data[:,1] = (torch.sum(Y[:,:,1]*H_re+Y[:,:,0]*H_im, dim=0)/
                                    (data_noise_power+torch.sum(H_re**2+H_im**2, dim=0)))
                      
                    err = det_data-torch.Tensor([1.,0.])    
                    err_data = err_data+torch.sum(err**2)
    
                loss_current = err_data/(N_data_sym*N_used)
                loss = loss+SNRscaleFactor**SNR*loss_current        
            
    return loss/(len(SNR_L)*len(inds))

def test_recover_zero():
    loss = getLoss({}, 
            lossVersion='detector',
            inds=[1,2], 
            SNR_L=[-5], 
            seed=None, 
            max_iter=7, 
            ml_version='trivial', 
            SNRscaleFactor=1.,
            scen=scen0)
    assert np.allclose(float(loss), 1)  
 
            
def testGetLoss():
    ml = {'softm_main_scale': torch.tensor(2., requires_grad=True), 
          'softm_power': torch.tensor(1., requires_grad=True),
          'softm_SNR': torch.tensor(-0.05, requires_grad=True),
          }
    loss = getLoss(ml, ml_version=0, lossVersion='detector', 
                   inds=[1], SNR_L=[-12], seed=0)
    print ('Loss:', loss)
    loss.backward()
    
    lrate = 1e0
    print ('Gradient:')
    for key in ml.keys():
        print (key, ':', float(ml[key].grad))
        ml[key].data = ml[key].data-lrate*ml[key].grad
        ml[key].grad.zero_()
    
    loss = getLoss(ml, ml_version=0, inds=[1], lossVersion='detector', 
                   SNR_L=[-12], seed=0)
    print ('New loss:', loss)
          
   
        
def demoGradDescent(ml_version=5, 
                    lossVersion='detector', # 'detector' or 'relError'
                    trainTestSplitType=1, 
                    max_iter=3, 
                    GD_steps=200,
                    seed=None, 
                    lrate = 1e-2,
                    SNR_L=[-12]):
    global Leg
    if ml_version in ['ref', 'trivial']:
        ml = {}
    elif ml_version == 0:
        ml = {'softm_main_scale': torch.tensor(2.1, requires_grad=True), 
              'softm_power': torch.tensor(1.5, requires_grad=True),
              'softm_SNR': torch.tensor(-0.3, requires_grad=True),
              }
    elif ml_version == 0.5:
        ml = {'softm_main_scale': torch.tensor(2.36, requires_grad=True), 
              'softm_power': torch.tensor(1.51, requires_grad=True),
              'softm_SNR': torch.tensor(-0.32, requires_grad=True),
              'sigm_power': torch.tensor(0.43, requires_grad=True),
              'sigm_SNR': torch.tensor(-0.067, requires_grad=True),
              }
        ml = {'softm_main_scale': torch.tensor(2.36, requires_grad=True), 
              'softm_power': torch.tensor(1.541, requires_grad=True),
              'softm_SNR': torch.tensor(-0.12, requires_grad=True),
              'sigm_power': torch.tensor(0.433, requires_grad=True),
              'sigm_SNR': torch.tensor(-0.028, requires_grad=True),
              }        
    elif ml_version == 0.7:
        ml = {'softm_main_scale': torch.tensor(2.36, requires_grad=True), 
              'softm_power': torch.tensor(1.541, requires_grad=True),
              'softm_SNR': torch.tensor(-0.119, requires_grad=True),
              'softm_SNR1': torch.tensor(0.0054, requires_grad=True),
              'sigm_power': torch.tensor(0.433, requires_grad=True),
              'sigm_SNR': torch.tensor(-0.0276, requires_grad=True),
              'sigm_SNR1': torch.tensor(0.0039, requires_grad=True),
              }
    elif ml_version == 0.8:
        ml = {'softm_main_scale': torch.tensor(2.36, requires_grad=True), 
              'softm_power': torch.tensor(1.541, requires_grad=True),
              'softm_SNR': torch.tensor(-0.119, requires_grad=True),
              'softm_SNR1': torch.tensor(0.0054, requires_grad=True),
              'sigm_power': torch.tensor(0.433, requires_grad=True),
              'sigm_SNR': torch.tensor(-0.0276, requires_grad=True),
              'sigm_SNR1': torch.tensor(0.0039, requires_grad=True),
              'common_power_sigm': torch.tensor(0.0, requires_grad=True),
              'common_power_sigm_SNR': torch.tensor(0.0, requires_grad=True),
              'common_power_softm': torch.tensor(0.0, requires_grad=True),
              'common_power_softm_SNR': torch.tensor(0.0, requires_grad=True),
              }
    elif ml_version == 1:
        ml = {'softm_main_scale': torch.tensor(2.37, requires_grad=True), 
              'softm_power': torch.tensor(1.52, requires_grad=True),
              'softm_SNR': torch.tensor(-0.31, requires_grad=True),
              'sigm_power': torch.tensor(0.42, requires_grad=True),
              'sigm_SNR': torch.tensor(-0.07, requires_grad=True),
              'bayes_c0': torch.tensor(0.03, requires_grad=True),
              'bayes_c1': torch.tensor(0.6, requires_grad=True),
              }
    elif ml_version == 2:
        ml = {'softm_main_scale': torch.tensor(2.1, requires_grad=True), 
              'softm_power': torch.tensor(1.5, requires_grad=True),
              'softm_SNR': torch.tensor(-0.3, requires_grad=True),
              'sigm_power': torch.tensor(0., requires_grad=True),
              'sigm_SNR': torch.tensor(-0.3, requires_grad=True),
              'bayes_c0': torch.tensor(0.04, requires_grad=True),
              'bayes_c1': torch.tensor(0.7, requires_grad=True),
              'A0': torch.ones((scen0.Nrx, scen0.Nrx), requires_grad=True),
              'A1': torch.ones((scen0.Nrx, scen0.Nrx), requires_grad=True),
              'softm_power1': torch.tensor(1.5, requires_grad=True),
              'softm_SNR1': torch.tensor(-0.3, requires_grad=True),
              'sigm_power1': torch.tensor(0., requires_grad=True),
              'sigm_SNR1': torch.tensor(-0.3, requires_grad=True),
              }
    elif ml_version == 3:
        ml = {'softm_main_scale': torch.tensor(2.1, requires_grad=True), 
              'bayes_pos1': torch.tensor(0.01, requires_grad=True), 
              'bayes_pos2': torch.tensor(0.4, requires_grad=True),
              'softm_power': torch.tensor(1.5, requires_grad=True),
              'softm_SNR': torch.tensor(-0.3, requires_grad=True),
              'sigm_power': torch.tensor(0., requires_grad=True),
              'sigm_SNR': torch.tensor(-0.3, requires_grad=True),
              'bayes_c0': torch.tensor(0.04, requires_grad=True),
              'bayes_c1': torch.tensor(0.7, requires_grad=True),
              }
    elif ml_version == 4:
        ml = {'softm_main_scale': torch.tensor(2.1, requires_grad=True), 
              'coeff_current_iter': torch.tensor(0.1, requires_grad=True), 
              'softm_power': torch.tensor(1.5, requires_grad=True),
              'softm_SNR': torch.tensor(-0.3, requires_grad=True),
              'sigm_power': torch.tensor(0., requires_grad=True),
              'sigm_SNR': torch.tensor(-0.3, requires_grad=True),
              'bayes_c0': torch.tensor(0.04, requires_grad=True),
              'bayes_c1': torch.tensor(0.7, requires_grad=True),
              }
    if ml_version == 5:
        ml = {'softm_main_scale': torch.tensor(2.1, requires_grad=True), 
              'softm_power': torch.tensor(1.5, requires_grad=True),
              'softm_power_diff': torch.tensor(0.0, requires_grad=True),
              'softm_SNR': torch.tensor(-0.3, requires_grad=True),
              'softm_SNR_diff': torch.tensor(-0.0, requires_grad=True),
              'sigm_power': torch.tensor(0., requires_grad=True),
              'sigm_power_diff': torch.tensor(0.0, requires_grad=True),
              'sigm_SNR': torch.tensor(-0.3, requires_grad=True),
              'sigm_SNR_diff': torch.tensor(-0.0, requires_grad=True),
              'bayes_c0': torch.tensor(0.04, requires_grad=True),
              'bayes_c1': torch.tensor(0.7, requires_grad=True),
              }
    elif ml_version == 6:
        ml = {'softm_main_scale': torch.tensor(2.36, requires_grad=True), 
              'softm_power': torch.tensor(1.541, requires_grad=True),
              'softm_SNR': torch.tensor(-0.119, requires_grad=True),
              'softm_SNR1': torch.tensor(0.0054, requires_grad=True),
              'sigm_power': torch.tensor(0.433, requires_grad=True),
              'sigm_SNR': torch.tensor(-0.0276, requires_grad=True),
              'sigm_SNR1': torch.tensor(0.0039, requires_grad=True),
              'softm_SNR_scale': torch.tensor(0.0, requires_grad=True),
              'max_power_scale': torch.tensor(0.0, requires_grad=True),
              'max_power_sigm': torch.tensor(0.0, requires_grad=True),
              'max_power_softm': torch.tensor(0.0, requires_grad=True),              
              }
    elif ml_version == 7:
        ml = {'softm_main_scale': torch.tensor(2.36, requires_grad=True), 
               'softm_power': torch.tensor(1.541, requires_grad=True),
               'softm_SNR': torch.tensor(-0.119, requires_grad=True),
               'softm_SNR1': torch.tensor(0.0054, requires_grad=True),
               'sigm_power': torch.tensor(0.433, requires_grad=True),
               'sigm_SNR': torch.tensor(-0.0276, requires_grad=True),
               'sigm_SNR1': torch.tensor(0.0039, requires_grad=True),
               }

    elif ml_version in [8, 12]: 
        ml =   {"bayes_exp_c0": tensor(-0.0094, requires_grad=True), 
 "bayes_exp_c1": tensor(0.0176, requires_grad=True),
 "bayes_sigm_c0": tensor(-0.0006,requires_grad=True), 
 "bayes_sigm_c1": tensor(0.0104, requires_grad=True),
 "bayes_softm_c0":tensor( -5.9802e-10,requires_grad=True),
 "bayes_softm_c1": tensor(-0.0389,requires_grad=True),
 "max_power_scale": tensor(-0.0369, requires_grad=True),
 "max_power_sigm": tensor(-0.0072, requires_grad=True),
 "max_power_softm": tensor(0.0036,requires_grad=True),
 "sigm_SNR": tensor(-0.0140, requires_grad=True),
 "sigm_SNR1": tensor(0.0013, requires_grad=True),
 "sigm_power": tensor(0.4305, requires_grad=True),
 "softm_SNR": tensor(-0.1328, requires_grad=True),
 "softm_SNR1": tensor(0.0119, requires_grad=True),
 "softm_SNR_scale": tensor(0.0183, requires_grad=True),
 "softm_main_scale": tensor(3.8476, requires_grad=True),
 "softm_power": tensor(1.5435, requires_grad=True)}



    elif ml_version == 9:
        ml = {'bayes_exp_c0': tensor(0.0020, requires_grad=True),
             'bayes_exp_c1': tensor(-0.0036, requires_grad=True),
             'bayes_sigm_c0': tensor(1.4152e-07, requires_grad=True),
             'bayes_sigm_c1': tensor(3.6723e-06, requires_grad=True),
             'bayes_softm_c0': tensor(-1.1423e-10, requires_grad=True),
             'bayes_softm_c1': tensor(-0.0094, requires_grad=True),
             'max_power_scale': tensor(0.0069, requires_grad=True),
             'max_power_sigm': tensor(0.0012, requires_grad=True),
             'max_power_softm': tensor(0.0011, requires_grad=True),
             'sigm_SNR': tensor(-0.0302, requires_grad=True),
             'sigm_SNR1': tensor(0.0314, requires_grad=True),
             'sigm_power': tensor(0.4333, requires_grad=True),
             'softm_SNR': tensor(-0.1221, requires_grad=True),
             'softm_SNR1': tensor(0.0092, requires_grad=True),
             'softm_SNR_scale': tensor(0.0146, requires_grad=True),
             'softm_main_scale': tensor(2.3631, requires_grad=True),
             'softm_power': tensor(1.5414, requires_grad=True),
             'power_diff_exp': tensor(0., requires_grad=True),
             'power_diff_sigm': tensor(0., requires_grad=True),
             'power_diff_softm': tensor(0., requires_grad=True),
             'power_diff_exp': tensor(0., requires_grad=True),
             'max_p_ant_exp': tensor(0., requires_grad=True),
             'max_p_ant_sigm': tensor(0., requires_grad=True),
             'max_p_ant_softm': tensor(0., requires_grad=True),
             }
        
        
    elif ml_version == 'poly0':
        deg = 40
        N_used = scen0.RB_num*scen0.RB_size
        Leg = getLegendre(N_used, deg)
        ml = {'deg_': deg,
              'c0': torch.zeros((deg,), requires_grad=True), 
              'c1': torch.zeros((deg,), requires_grad=True),
              }        

    
    if trainTestSplitType == 0:
        trainInds = range(1,141,2)
        testInds = range(2,141,2)
    elif trainTestSplitType == 1:
        trainInds = range(1,71)
        testInds = range(71,141)  
    elif trainTestSplitType == 2:
        trainInds = range(1,141)
        testInds = range(1,141)        
           
    print ('---- Old param vals:')
    pp = pprint.PrettyPrinter()
    pp.pprint(ml)   
    
    loss_train = getLoss(ml, inds=trainInds, SNR_L=SNR_L, 
                         lossVersion=lossVersion,
                         seed=seed, ml_version=ml_version, max_iter=max_iter)
    #loss_test = getLoss(ml, inds=testInds, SNR_L=SNR_L, 
    #                    lossVersion=lossVersion,
    #                    seed=seed, ml_version=ml_version, max_iter=max_iter)
    print ('\nInitial loss on train:', float(loss_train))
    #print ('Initial loss on test:', float(loss_test))
    final_loss = 1000
    t0 = time.time()
    for k in range(GD_steps):
        print ('\n==== GD step', k)
        if len(ml) > 0:      
            loss_train.backward()
    
        for key in ml.keys(): 
            if key.endswith('_'):
                continue
            ml[key].data = ml[key].data-lrate*ml[key].grad
            ml[key].grad.zero_()
            
        
        
        loss_train = getLoss(ml, inds=trainInds, SNR_L=SNR_L, 
                             lossVersion=lossVersion,
                             seed=seed, ml_version=ml_version, max_iter=max_iter)
        if loss_train.data.numpy() < final_loss:
            final_loss = float(loss_train.data.numpy())
            final_grad = ml
        
        print ('Loss on train:', float(loss_train))   
        loss_test = getLoss(ml, inds=testInds, SNR_L=SNR_L, 
                            lossVersion=lossVersion,
                            seed=seed, ml_version=ml_version, max_iter=max_iter)
        print ('Loss on test:', float(loss_test))      
        print ('\nNew param vals:')
        pp.pprint(ml)
        
    print('=============')
    print('Final Loss:', final_loss)
    print('Final grad:', final_grad)
    print('=============')
    #print ('\nNew param vals:')
    #pp.pprint(ml)
            
    print ('\nTime elapsed:', time.time()-t0)
    
    
        

if __name__ == "__main__":
    demoGradDescent(ml_version=8,
                    lossVersion='detector', # or 'relError'
                    trainTestSplitType=0, 
                    max_iter=3, 
                    GD_steps=200,
                    seed=4, 
                    lrate = 1e-2,
                    SNR_L=[-12])
    


