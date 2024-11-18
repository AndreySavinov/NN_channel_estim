% Channel generator according to open-source Quadriga channel
clear all;
close all;
clc;
addpath('./quadriga_src')
%RandStream.setGlobalStream(RandStream('mt19937ar', 'seed', 1234567));
RandStream.setGlobalStream(RandStream('mt19937ar', 'seed', 3000000));

%% Main sim parametrs
Tx_mode = 'NewRadio'; % NewRadio or LTE
ChanParam.N_tti = 100; % number of TTI for simulation
ChanParam.RB_num = 16; % Resourse block num 4, 16

% consider two cfg :
% type1_64Tx_1Rx - 64 ant at BS, 1 ant at UE
% type2_4Tx_1Rx - 4 ant at BS, 1 ant at UE

ChanParam.ueChanType                = 'type1_64Tx_1Rx'; % type1_64Tx_1Rx, type2_4Tx_1Rx
%ChanParam.ChannelScen               = '3GPP_3D_UMa_NLOS'; % 3GPP_3D_UMa_NLOS
ChanParam.ChannelScen               = 'DRESDEN_UMa_LOS'; % 3GPP_3D_UMa_NLOS
% avaliable scenarios lie in the folder ./quadriga_src/config

ChanParam.log_sim_string = ['SimCFG-Tx-mode=', Tx_mode, '-TTInum=', num2str(ChanParam.N_tti), ...
    '-AntCFG=', ChanParam.ueChanType, '-Scenario=', ChanParam.ChannelScen, ...
    '-RBnum=', num2str(ChanParam.RB_num)];
fprintf('%s\n', ChanParam.log_sim_string)

%% ------------------------------------ Other Channel settings -------------------------------------------
step_ue = 5.0; % that parameter control how fat users separated
sp = 5.0; % user velocity in km/h
ChanParam.gen_frequency_response = true; % generated required frequency response at the end of simulation

ChanParam.NumUe                     = 1;      % number of UE
ChanParam.NumBs                     = 1;      % BS number 
ChanParam.Nsec                      = 1;      %Sectors of BS number 
ChanParam.AnalogBFWeights           = [];
ChanParam.carrierFreqDL             = 3.5e9;  % Hz
ChanParam.carrierFreqUL             = 3.5e9;  % Hz

ChanParam.eNbHeight                 = 25;     % meters
ChanParam.ISD                       = 500;    % Interside distance [m]
ChanParam.ueMaxDist                 = 0.5*ChanParam.ISD;  % meters
ChanParam.ueMinDist                 = 50;     % meters
ChanParam.ueMaxAzimuth              = 60;     % degrees
ChanParam.ueMinHeight               = 1.5;    % meters
ChanParam.ueMaxHeight               = 1.5;    % meters
ChanParam.ueSpeed                   = sp;     % [km/h]
ChanParam.VerticalSpacing           = 0.9;    % wavelength
ChanParam.HorizontalSpacing         = 0.5;    % wavelength
ChanParam.Tilt                      = 16;     % degrees
ChanParam.GetULch                   = 0;
ChanParam.NumCentr                  = 1;
ChanParam.UeGen                     ='By-Cell-Centroids'; % (By-Cell, By-Site, By-Cell-Centroids)
ChanParam.PhaseOffset               =5/180*pi;
ChanParam.CentRadius                =1;

ChanParam.step_ue = step_ue;

ChanParam.N_sc_rb = 12; % number of subcarriers per RB

ChanParam.N_ofdm_tti = 14; % number of OFDM symbols per TTI
if strcmp(Tx_mode, 'NewRadio')
    ChanParam.f_space = 30e3; % in case of NewRadio scenario frequency spacing is 30kHz
    %ChanParam.T_tti = 0.5e-3; % TTI time including CP is 0.5ms
    ChanParam.T_tti = ChanParam.N_ofdm_tti/ChanParam.f_space; % % this is length of TTI without CP     
elseif strcmp(Tx_mode, 'LTE') 
    ChanParam.f_space = 15e3; % in case of LTE spacing is 15kHz
    %ChanParam.T_tti = 1e-3; % TTI time including CP is 1ms
    ChanParam.T_tti = ChanParam.N_ofdm_tti/ChanParam.f_space; % this is length of TTI without CP 
else
    error('Not supported Tx-mode, should be LTE or NewRadio');
end


ChanParam.channelSnapshotInterval   =  ChanParam.T_tti / ChanParam.N_ofdm_tti;   % 1 snapshot [seconds]

getFDDch_Qr20(ChanParam);
