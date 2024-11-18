


function [CE_DATA]  = CE_TTI_client(IN_DATA,estimationParams, ML_coef, PHY_param)

N_pilot_sym=2;

SRS_transform_matrix=IN_DATA.SRS_transform_matrix;

switch estimationParams.beam_transform
    
    case 1
        
        h_beam_noisy=IN_DATA.h_beam_noisy;
        % N_ports -> 64 antennas
        for i=1:N_pilot_sym
            h2_f_noisy(:,:,i)=(squeeze(h_beam_noisy(:,:,i))*SRS_transform_matrix.').';
        end
        
    case 0
        
        h2_f_noisy=IN_DATA.h_f_noisy;
end

%SERVER_HOST='http://10.2.1.1:8082/CE_TTI';
SERVER_HOST='http://10.2.2.3:8081/CE_TTI';
SERVER_HOST='http://10.30.101.210:8080/CE_TTI'; 

if ~isfield(estimationParams,'SNR')
    estimationParams.SNR = 0;
end
if ~isfield(estimationParams,'N_TTI')
    estimationParams.N_TTI = 1;
end
if ~isfield(estimationParams,'UE_indx')
    estimationParams.UE_indx = 1;
end
 if ~isfield(estimationParams,'UE_number')
    estimationParams.UE_number = 1;
end
 if ~isfield(estimationParams,'Nfft')
    estimationParams.Nfft = 512;
 end

PHY_param.Nfft=512;

json_data = struct('SNR',estimationParams.SNR,'RB_num',estimationParams.RB_num,...
                   'N_TTI',estimationParams.N_TTI,'N_pilot',estimationParams.N_pilot,...
                  'UE_indx',estimationParams.UE_indx,'UE_number',estimationParams.UE_number,...
                  'Nrx',estimationParams.Nrx,'RB_size',estimationParams.RB_size,...
                  'Nfft',PHY_param.Nfft,...
                  'h_f_noisy_re',real(h2_f_noisy),'h_f_noisy_im',imag(h2_f_noisy));
              
options = weboptions('MediaType','application/json','Timeout',5);
response = webwrite(SERVER_HOST,json_data,options);

[h_pilots_f, h_data_f] = recover_result_matrices_from_json(response);

SRS_beam_amplitudes=squeeze(h_data_f(:,:,1)).'*conj(SRS_transform_matrix);
CE_DATA.SRS_beam_amplitudes=repmat(SRS_beam_amplitudes,1,1,12);

CE_DATA.SNR=estimationParams.SNR_dummy;
CE_DATA.h_data_recovered_f=h_data_f;
CE_DATA.h_pilots_recovered_f=h_pilots_f;

end



function [h_pilots_f,h_data_f] = recover_result_matrices_from_json(result_json)

pilots_json = result_json.h_f_recovered_pilots;
data_json = result_json.h_f_recovered_data;


h_pilots_f = complex(pilots_json(:,:,:,1),pilots_json(:,:,:,2));
h_data_f = complex(data_json(:,:,:,1),data_json(:,:,:,2));
end

