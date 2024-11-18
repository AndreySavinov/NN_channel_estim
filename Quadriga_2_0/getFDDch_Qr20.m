function [Cir_DL,Cir_UL,ChanParam]=getFDDch_Qr20(ChanParam)

%---------------------------------------------------------------------------------------------------
% if (ChanParam.NumBs>1)
%   ChanParam.Nsec=3;
% else
%   ChanParam.Nsec=1;
% end
%---------------------------------------------------------------------------------------------------
if strcmp(ChanParam.ueChanType,'type01_02x2')
  %eNB side
  Nr_tx=4;
  Nc_tx=1;
  ChanParam.NumAntEl_BS=Nc_tx*Nr_tx*2;
  %ue side
  Nr_rx=1;
  Nc_rx=1;
  ChanParam.NumAntEl_UE=Nc_rx*Nr_rx*2;
  
elseif strcmp(ChanParam.ueChanType,'type01_04x2')
  %eNB side
  Nr_tx=4;
  Nc_tx=2;
  ChanParam.NumAntEl_BS=Nc_tx*Nr_tx*2;
  %ue side
  Nr_rx=1;
  Nc_rx=1;
  ChanParam.NumAntEl_UE=Nc_rx*Nr_rx*2;
    
elseif strcmp(ChanParam.ueChanType,'type01_04x4')
  %eNB side
  Nr_tx=4;
  Nc_tx=2;  
  ChanParam.NumAntEl_BS=Nc_tx*Nr_tx*2;
  %ue side
  Nr_rx=1;
  Nc_rx=2;  
  ChanParam.NumAntEl_UE=Nc_rx*Nr_rx*2;
elseif strcmp(ChanParam.ueChanType,'type01_04x8')
  %eNB side
  Nr_tx=4;
  Nc_tx=8;  
  ChanParam.NumAntEl_BS=Nc_tx*Nr_tx*2;
  %ue side
  Nr_rx=1;
  Nc_rx=2;  
  ChanParam.NumAntEl_UE=Nc_rx*Nr_rx*2;  
  
elseif strcmp(ChanParam.ueChanType,'type1_64Tx_1Rx')
  
    Nr_tx=4;
    Nc_tx=8;  
    ChanParam.NumAntEl_BS=Nc_tx*Nr_tx*2;
    %ue side
    Nr_rx=1;
    Nc_rx=1;  
    ChanParam.NumAntEl_UE=Nc_rx*Nr_rx*2;  
    
elseif strcmp(ChanParam.ueChanType,'type2_4Tx_1Rx')
    
    %eNB side
    Nr_tx=2;
    Nc_tx=1;
    ChanParam.NumAntEl_BS=Nc_tx*Nr_tx*2;
    
    %ue side
    Nr_rx=1;
    Nc_rx=1;
    ChanParam.NumAntEl_UE=Nc_rx*Nr_rx*2;      
end

%antenna groups for analog BF and rearange indexes similar to airview
AntGr=zeros(2*Nc_tx,Nr_tx,ChanParam.Nsec);
AntHpol=1:2:(2*Nc_tx*Nr_tx*ChanParam.Nsec);
AntVpol=2:2:(2*Nc_tx*Nr_tx*ChanParam.Nsec);
AntHpol=reshape(AntHpol,[Nr_tx Nc_tx ChanParam.Nsec]);
AntVpol=reshape(AntVpol,[Nr_tx Nc_tx ChanParam.Nsec]);
AntGr(1:Nc_tx,:,:)=permute(AntHpol,[2 1 3]);
AntGr(Nc_tx+1:end,:,:)=permute(AntVpol,[2 1 3]);

if isempty(ChanParam.AnalogBFWeights)  
  ChanParam.AnalogBFWeights=ones(1,Nr_tx)*1/sqrt(Nr_tx);
end

% =====================================================================
% Create simulation parameters structure
% =====================================================================
% addpath('source');
simPar = qd_simulation_parameters();
simPar.center_frequency = ChanParam.carrierFreqDL;
% simPar.use_random_initial_phase = 0; 
simPar.set_speed (ChanParam.ueSpeed,ChanParam.channelSnapshotInterval);
simPar.use_absolute_delays = 0;                              % Include delay of the LOS path
simPar.show_progress_bars = 1;                               % Disable progress bars

% =====================================================================
% Set layout
% =====================================================================  
 simLayout = qd_layout( simPar );
% -------Design TX array (BS) -------------------------------------------
 antTX  = qd_arrayant( '3gpp-3d',  Nr_tx, Nc_tx, simPar.center_frequency, 3);
 antTX.element_position(3,:)=antTX.element_position(3,:) * (ChanParam.VerticalSpacing / ChanParam.HorizontalSpacing);
 simLayout.tx_array = antTX;
% -----------------------------------------------------------------------
% ------- Position & Tild -----------------------------------------------
 simLayout.tx_position(3) = ChanParam.eNbHeight; % X m Tx height 
 simLayout.tx_array.rotate_pattern(ChanParam.Tilt,'y');
% -----------------------------------------------------------------------
% ------- Design RX array (BS) -------------------------------------------
if strcmp(ChanParam.ueChanType, 'type1_64Tx_1Rx') || strcmp(ChanParam.ueChanType, 'type2_4Tx_1Rx')
    antRX  = qd_arrayant( '3gpp-3d',  Nr_rx, Nc_rx, simPar.center_frequency, 1); % one omnidirectional antenna on user
else
    antRX  = qd_arrayant( '3gpp-3d',  Nr_rx, Nc_rx, simPar.center_frequency, 6); % cross polarized antennas at least 2 on user
end
 simLayout.rx_array = antRX;
 simLayout.rx_position = [20; 0; ChanParam.ueMaxHeight]; % X m Tx height 
  
%  simLayout.rx_position(1:2,:)=getUExy( ChanParam.NumUe,ChanParam.NumCentr,ChanParam.Nsec,...
%   ChanParam.UeGen,simLayout.tx_position(1:2,:),ChanParam.ueMaxDist,ChanParam.ueMinDist,...
%   ChanParam.PhaseOffset,ChanParam.CentRadius );
% -----------------------------------------------------------------------
 
%UMal = 'BERLIN_UMa_LOS';                                % LOS scenario name
%UMan = 'BERLIN_UMa_NLOS';                               % NLOS scenario name
simLayout2 = copy (simLayout);
%% ------------- Number of UE and their positions ------------------------
simLayout.no_rx = ChanParam.NumUe;
ueTrackLen = ChanParam.ueSpeed/3.6 * ChanParam.N_tti *ChanParam.N_ofdm_tti * ChanParam.channelSnapshotInterval;
for i_rx=1:simLayout.no_rx    
 %% ---------- Users positions on the track ----------------------------------   
  simLayout.track(i_rx)       = qd_track('linear',ueTrackLen,pi/8);     % Linear track, ueTrackLen [m] length by angle pi/8 (north-east)
  simLayout.track(i_rx).name  = ['Rx',num2str(i_rx,'%04.0f')];
  if ueTrackLen>0      
    simLayout.track(i_rx).interpolate_positions( simLayout.simpar.samples_per_meter );
    simLayout.track(i_rx).compute_directions;
  end
  simLayout.track(i_rx).initial_position  = [15; -10+ChanParam.step_ue*(i_rx-1); ChanParam.ueMaxHeight];   % Same start point  
  simLayout.track(i_rx).segment_index     = [1]; %fix([1/simLayout.track(i_rx).no_snapshots, 0.5, 0.75]*simLayout.track(i_rx).no_snapshots);      % Segments indexes in snapshots points (all snapshots = 20m*120[densety]*)  
%% ---------------------- Scenarios by segments ---------------------
%   simLayout.track(i_rx).scenario = { UMal, UMal, UMan };            % Scenarios of segments;
   simLayout.track(i_rx).scenario = { ChanParam.ChannelScen};            % Scenarios of segments;
end

if 0
  simLayout.visualize ([],[],0);  
end
% =====================================================================
% Create parameter set
% =====================================================================
paramSet = simLayout.init_builder;  
  
% =====================================================================
% Generate channel
% =====================================================================
paramSet.gen_ssf_parameters;
simPar.use_spherical_waves=1;   % switch 'On' Drifting
H_dl  = get_channels(paramSet);
H_dl2 = merge(H_dl);

H.QrigaChannel = H_dl2; 
H.QrigaParams  = paramSet;

% time domain channel is saved here
file_name_chan_time = [ChanParam.log_sim_string, '.mat'];
save(file_name_chan_time,'H');

fprintf('Channel Generated sucessfully and saved in file\n%s\n', file_name_chan_time);

% H - is Quadriga channel object, it is more easier to save it and then
% generate frequency response with foloowing code

%% ----- Resample channel according Bandwidth --------------------------- 
% obtain frequency response

if ChanParam.gen_frequency_response
    % example of access to frequency domain channel
    BW = ChanParam.RB_num * ChanParam.N_sc_rb * ChanParam.f_space;
    SC_size = ChanParam.RB_num * ChanParam.N_sc_rb;

    for i_rx=1:simLayout.no_rx   % loop over user number
        Htmp = H_dl2(i_rx).fr( BW, SC_size);     % Freq.-domain channel    
        Hfr{i_rx} = Htmp(:,:,:,1:end-1); % Nue_ant x Nbs_ant x BW_rb x OFDM_idx
    end

    %Hfr{i} - is frequency channel for user-i with dimensions Nue_ant x Nbs_ant x BW_rb x OFDM_idx
    file_name_chan_freq =  [ChanParam.log_sim_string, '-freq.mat'];
    save(file_name_chan_freq,'Hfr');
    fprintf('Frequency response saved in file\n%s\n', file_name_chan_freq);
else
    fprintf('Channel frequency response was not saved due to flag ChanParam.gen_frequency_response=false');    
end


