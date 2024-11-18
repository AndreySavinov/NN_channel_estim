Main file is GENERATE_CHANNEL.m

Unzip quadriga_src before first usage
 
To change channel cgf use ChanParam.ChannelScen, avaliable channels are in ./quadriga_src/config (e.g. ChanParam.ChannelScen='3GPP_3D_UMa_NLOS')

OutPut of the running main is two .mat files:

first is devoted to time impulse response of channel and all parametrs
second with suffix -freq.mat only frequency response is stored in cell array Hfr

example how frequency response is obtained is in function getFDDch_Qr20(ChannelParam)

number of cells in cell array Hfr is number of users

H1 = Hfr{1} - response from first user in format N_ue_ant x N_bs_ant x BW_sc x OFDM_idx

You can choose number of TTI for simulation by changing
ChanParam.N_tti

RB number control:
ChanParam.RB_num=4, ChanParam.RB_num=16

Two antenna configurations required to test:
ChanParam.ueChanType = 'type1_64Tx_1Rx' - 64 ant BS, 1 ant UE
ChanParam.ueChanType = 'type2_4Tx_1Rx' - 4 ant BS, 1 ant UE

There are two transmission cfg to check:
Tx_mode='NewRadio' - new radio numberology f_space = 30kHz
Tx_mode='LTE' - LTE numerology f_space = 15kHz

Other parametrs may not be changed

Output channel is not normalized on path loss

Comments:

Each TTI consist here from 14OFDM symbols, simulation time is measured in TTI's because within TTI codeword is transmitted

In order to get Channel for specific TTI=5 for UE=1 you need to do following:
H_tmp = Hfr{1};
N_ofdm = 14;
tti_idx = 5; 
H_tti = squeeze(H_tmp(:,:,:, (tti_idx-1)*N_ofdm + 1 : tti_idx*N_ofdm));




