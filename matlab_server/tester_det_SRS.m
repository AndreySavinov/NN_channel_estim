function [err_power_data] = tester_det_SRS(scenario, estimator_DMRS, estimator_SRS, ML_coef1)

seed = 100*scenario.seed+10000*scenario.index;
rng(seed);

% N-of-Rx antennas
Nrx = scenario.Nrx;

comb = scenario.comb;

% N-of-Users
UE_number = scenario.UE_number;

% RB_size = 12 subcarriers
RB_size=scenario.RB_size;

RB_num=scenario.RB_num;

% N-of-subcarriers (max=600)
N_used=RB_num*RB_size; % var - use it to change simulation set   

% N of symbols per TTI
N_ofdm = 14;

% N-of-TTIs
N_TTI=scenario.N_TTI;

N_data_sym=12*N_TTI;

% N-of-pilot-symbols
N_pilot_sym = 2*N_TTI;

% Pilot positions in TTI
pilot_positions=[4 11];

% Data positions in TTI
data_positions1=[1:3];
data_positions2=[5:10];
data_positions3=[12:14];

% Constelation order
QAM_order=6;
QAM_points=2^QAM_order;

SNR = scenario.SNR;

UE_indx=scenario.UE_indx;

index=num2str(scenario.index);

% %load channel
% load(['SimCFG-Tx-mode=NewRadio-TTInum=100-AntCFG=type1_64Tx_1Rx-Scenario=DRESDEN_UMa_NLOS-RBnum=16-freq' '_seed' index]);
% H_tmp = Hfr{1} ;
% H_new=H_tmp(:,:,:,1:N_ofdm*N_TTI);
% seed=num2str(scenario.seed+40);
% save(['temp_chan' '_seed' seed],'H_new');

%fast channel loading
load(['temp_chan' '_seed' index]);
H_tmp=H_new;

% z = size(H_full_set(scenario.index,:,:,:,:));
% H_tmp = reshape(H_full_set(scenario.index,:,:,:,:),[z(2:end) 1]);


h_pilot=zeros(UE_number,Nrx,N_used,N_pilot_sym);
% Extract pilot symbols
for tti_idx=1:N_TTI
    h_pilot(:,:,1:(1+comb):N_used,2*(tti_idx-1)+1) = (sqrt(1+comb))*H_tmp(1,1:Nrx,1:(1+comb):N_used, (tti_idx-1)*N_ofdm+pilot_positions(1));
    h_pilot(:,:,1:(1+comb):N_used,2*(tti_idx-1)+2) = (sqrt(1+comb))*H_tmp(1,1:Nrx,1:(1+comb):N_used, (tti_idx-1)*N_ofdm+pilot_positions(2));
end


% Extract data symbols
for tti_idx=1:N_TTI
    h_data(:,:,1:N_used,12*(tti_idx-1)+data_positions1-0) = H_tmp(1,1:Nrx,1:N_used, (tti_idx-1)*N_ofdm+data_positions1);
    h_data(:,:,1:N_used,12*(tti_idx-1)+data_positions2-1) = H_tmp(1,1:Nrx,1:N_used, (tti_idx-1)*N_ofdm+data_positions2);
    h_data(:,:,1:N_used,12*(tti_idx-1)+data_positions3-2) = H_tmp(1,1:Nrx,1:N_used, (tti_idx-1)*N_ofdm+data_positions3);
end

% calculate averaged UE power
UE_power=mean( mean( mean( squeeze(h_pilot(UE_indx,:,:,:)).*conj(squeeze(h_pilot(UE_indx,:,:,:))) )));

% generate white noise for DMRS pilots
white_noise_p=(randn(Nrx, N_used, N_pilot_sym)+1i*randn(Nrx, N_used, N_pilot_sym)) / sqrt(2);
noise_p=sqrt(mean(UE_power))*white_noise_p/sqrt(10^(SNR/10));

% generate channel response of single UE with noise
h_f_noisy=squeeze(h_pilot(UE_indx,:,:,:))+noise_p;

% dummy (depends on pilots scaling in comb mode)
h_f_noisy=h_f_noisy/sqrt(1+comb);

% estimate noise power per subcarrier
noise_power = sum(sum(sum(abs(noise_p).*abs(noise_p))))/(Nrx*N_used*N_pilot_sym); %   -  algorithm is required !!!!!

% generate white noise for data
white_noise_d=(randn(Nrx, N_used, N_data_sym)+1i*randn(Nrx, N_used, N_data_sym)) / sqrt(2);
noise_d=sqrt(mean(UE_power))*white_noise_d/sqrt(10^(SNR/10));

% Generate test data
data_tx = randi([0 1],N_used*QAM_order*N_TTI*12,1);
dataInMatrix = reshape(data_tx,length(data_tx)/QAM_order,QAM_order);   % Reshape data into binary k-tuples, k = log2(M)
dataSymbolsIn = bi2de(dataInMatrix);                                  % Convert to integers
s_tx = reshape( qammod(dataSymbolsIn,QAM_points,'UnitAveragePower', true) , N_used, N_data_sym );     % Gray coding, phase offset = 0;


% generate channel response of single UE with noise
for k=1:N_data_sym
    for m=1:N_used
        data_noisy(:,m,k)=squeeze(h_data(UE_indx,:,m,k))*s_tx(m,k)+squeeze(noise_d(:,m,k)).';
    end
end

% DMRS params
DMRS_Params.RB_size = RB_size;
DMRS_Params.RB_num = RB_num;
DMRS_Params.N_pilot = N_pilot_sym;
DMRS_Params.pilot_positions=pilot_positions;
DMRS_Params.comb=comb;
DMRS_Params.Nrx = scenario.Nrx;
DMRS_Params.N_ports = scenario.N_ports;
DMRS_Params.beam_transform = scenario.beam_transform;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SRS params
SRS_Params=DMRS_Params;
SRS_Params.RB_num=16;       % SRS is a wideband signal (32RB)
SRS_Params.comb = 0;
SRS_Params.N_ports = scenario.N_ports;

N_ports=scenario.N_ports;
comb_SRS=0;
gain_SRS=sqrt(2);  % SRS power is higher then DMRS
N_used_SRS=SRS_Params.RB_num*RB_size;

%fast channel loading
load(['SRS_chan' '_seed' index]);
H_tmp=H_new;

h_srs=zeros(UE_number,Nrx,N_used_SRS,N_pilot_sym);
% Extract srs symbols
for tti_idx=1:N_TTI
    h_srs(:,:,1:(1+comb_SRS):N_used_SRS,2*(tti_idx-1)+1) = (sqrt(1+comb_SRS))*H_tmp(1,1:Nrx,1:(1+comb_SRS):N_used_SRS, (tti_idx-1)*N_ofdm+pilot_positions(1));
    h_srs(:,:,1:(1+comb_SRS):N_used_SRS,2*(tti_idx-1)+2) = (sqrt(1+comb_SRS))*H_tmp(1,1:Nrx,1:(1+comb_SRS):N_used_SRS, (tti_idx-1)*N_ofdm+pilot_positions(2));
end

% generate white noise for SRS
white_noise_s=(randn(Nrx, N_used_SRS, N_pilot_sym)+1i*randn(Nrx, N_used_SRS, N_pilot_sym)) / sqrt(2);
noise_s=sqrt(mean(UE_power))*white_noise_s/sqrt(10^(SNR/10));

% generate channel response of single UE with noise
h_srs_noisy=gain_SRS*squeeze(h_srs(UE_indx,:,:,:))+noise_s;

% dummy (depends on pilots scaling in comb mode)
h_srs_noisy=h_srs_noisy/sqrt(1+comb);

% Beam angles estimation via SRS (100 TTIs delay between SRS and current DMRS)
[SRS_transform_matrix]=estimator_SRS(h_srs_noisy,SRS_Params,ML_coef1);

% Channel transfer to the beam domain
for i=1:N_pilot_sym
    % 64 antennas -> N_ports
    h_beam_noisy(:,:,i)=squeeze(h_f_noisy(:,:,i)).'*conj(SRS_transform_matrix);
    
%     % N_ports -> 64 antennas
%     h2_f_noisy(:,:,i)=(squeeze(h_beam_noisy(:,:,i))*SRS_transform_matrix.').';
end

% specify input data for Channel Estimation unit
IN_DATA.h_f_noisy=h_f_noisy;
IN_DATA.h_beam_noisy=h_beam_noisy;
IN_DATA.SRS_transform_matrix=SRS_transform_matrix;

%%% channel recovery %%%
CE_DATA = estimator_DMRS(IN_DATA, DMRS_Params, ML_coef1);

switch scenario.beam_transform
    case 1
        transform_matrix=SRS_transform_matrix;
        beam_amplitudes=CE_DATA.SRS_beam_amplitudes;
    case 0
        h_data_recovered_f=CE_DATA.h_data_recovered_f;
end

% %%% channel recovery (Huawei)
% h_beam_noisy_tmp = permute(h_beam_noisy, [2, 1, 3]);
% CE_DATA=estimator_DMRS(h_beam_noisy_tmp, DMRS_Params,ML_coef1);
% beam_amplitudes = permute(CE_DATA.h_data_recovered_f, [2, 1, 3]);

% Channel estimation
err_data=0;

% %%%%% v.2
% % Detection in DMRS beams basis
% % Only 8 ports are used and performance is better (gain is 0.05dB)
% transform_matrix=CE_DATA.DMRS_transform_matrix;
% beam_amplitudes = permute((CE_DATA.DMRS_beam_amplitudes), [2, 1, 3]);

%%%% v.3
% Detection in SRS beams basis (low complexity)

% error calculation
for k=1:N_data_sym
    for m=1:N_used

        switch scenario.beam_transform
            case 1
                H=squeeze(beam_amplitudes(m,:,k));
                Y=(squeeze(data_noisy(:,m,k))).'*conj(transform_matrix);
            case 0
                H=squeeze(h_data_recovered_f(:,m,k)).';
                Y=squeeze(data_noisy(:,m,k)).';
        end
        
        w_MMSE=pinv( noise_power+conj(H)*H.')*conj(H);
        det_data=Y*w_MMSE.';
        err=det_data-s_tx(m,k);
        err_data=err_data+sum(abs(err).^2);
    end
end



err_power_data=err_data/(N_used*N_data_sym);

   
