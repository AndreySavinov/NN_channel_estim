
function [beam_matrix]  = CE_TTI_det2_SRS(h2_f_noisy,estimationParams, ML_coef, PHY_param)


RB_size=estimationParams.RB_size;
RB_num=estimationParams.RB_num;
N_used=RB_num*RB_size;
N_pilot=estimationParams.N_pilot;
comb=estimationParams.comb;
Nrx=estimationParams.Nrx;

RB_set=[1 2 4 8 12 16];
[~,indx]=min(abs(RB_num-RB_set));
RB_tmp=RB_set(indx);

switch RB_tmp
    case 16 ; Nfft=1024;
    case 12 ; Nfft=1024; 
    case 8  ; Nfft=512; 
    case 4  ; Nfft=512; 
    case 2  ; Nfft=256; 
    case 1  ; Nfft=256;
end

Nzero=Nfft-N_used;
upsample_factor=1;
N_response=448*Nfft/2048;
N_shift = 16*Nfft/2048;  
max_N_peaks=PHY_param.max_N_peaks;


% Pilot positions in TTI
pilot_positions=estimationParams.pilot_positions;
gap=pilot_positions(2)-pilot_positions(1);

% Data positions in TTI
data_positions1=[1:3];
data_positions2=[5:10];
data_positions3=[12:14];

% generate SINC signal
SINC_f=circshift([ones(1,N_used) zeros(1,Nzero+(upsample_factor-1)*Nfft)],[0 -(N_used)/2]);
SINC_t=ifft(SINC_f)*sqrt(upsample_factor*Nfft);
SINC_scale=SINC_t(1);
SINC_t=SINC_t/SINC_scale;

% beam domain channel 
beam_amplitudes=zeros(max_N_peaks,N_used);
beam_transform_matrix=zeros(Nrx,max_N_peaks);

% N_ports=32;
% % generate DFT-based beams matrix
% W_pol = (dftmtx(2))';
% W_hor = (dftmtx(8))';
% W_vert = (dftmtx(4))';
% Wrez = kron(W_hor, W_vert);
% Wfin = kron(Wrez, W_pol);
% D = abs(diag(Wfin'*Wfin));
% Wfin = Wfin * diag(1.0./sqrt(D));
% 
% beam_power=zeros(1,Nrx);
% for p_idx = 1:N_pilot
%     tmp = squeeze( h2_f_noisy(:,:,p_idx) );
%     tmp = tmp.';
%     pwr = sum( abs(tmp*conj(Wfin)).^2 , 1);
%     beam_power = beam_power + pwr;
% end
% [~, beam_indx] = sort(beam_power, 'descend');
% 
% 
% beam_matrix=Wfin(:,beam_indx(1:N_ports));
% 
% % 64 antennas -> N_ports
% for p_idx = 1:N_pilot
%     tmp = squeeze( h2_f_noisy(:,:,p_idx) );
%     tmp = tmp.';
%     h_beam_noisy(:,:,p_idx)=(tmp*conj(beam_matrix)).';
% end
% 
% % N_ports -> 64 antennas 
% for p_idx = 1:N_pilot
%     tmp = squeeze( h_beam_noisy(:,:,p_idx) );
%     tmp = tmp.';
%     h2_f_noisy(:,:,p_idx)=(tmp*beam_matrix.').';
% end




if comb==1
    comb_matrix=ones(Nrx,N_used,N_pilot);
    comb_matrix(:,2:2:N_used,:)=zeros(Nrx,N_used/2,N_pilot);
    h2_f_noisy=h2_f_noisy.*comb_matrix;
end

    % dummy (depends on pilots scaling in comb mode)
    h2_f_noisy=h2_f_noisy*sqrt(1+comb);

    for j=1:Nrx
        for k=1:N_pilot
            
            h2_f_noisy_upsampled=(sqrt(1+comb))*circshift([squeeze(h2_f_noisy(j,:,k)) zeros(1,Nzero+(upsample_factor-1)*Nfft)],[0 -(N_used)/2]);
            data_array(j,k,:) = circshift(ifft(h2_f_noisy_upsampled)*sqrt(upsample_factor*Nfft),[0 N_shift]);

        end
    end
    
    N_pilot_svd = 2;
    sub_array_gain=1;


    for n=2:2:N_pilot
        
        if n==2
            switch Nrx
                case 64  
                    switch RB_tmp
                       case 16 ; ML=[25.2189  309.0313   16.6204   80.4742   1.4533    0.1593   1.8924   1.1876  0.7385  0.0784    0.8933   14.0418    0.0871]; %16RB 64RX
                       case 12 ; ML=[47.0774   47.6409   22.0680   73.5901   1.5863    0.1730   3.2818   0.7480  0.7678  0.0774    0.8885   16.7208    0.1018]; %12RB 64RX
                       case 8  ; ML=[33.3657  176.2515    8.8319   35.8971   1.9043    0.1776   2.2800   2.0769  1.1283  0.0591    0.8927    7.7038    0.0864]; % 8RB 64RX 
                       case 4  ; ML=[24.4985  133.2794    6.7398   35.0514   2.6703    0.2006   8.2865   0.6645  0.9128  0.1214    0.8785    4.7564    0.0268]; % 4RB 64RX
                       case 2  ; ML=[36.4611   27.3530    4.6128   19.4495   7.4382    0.3721  10.4721   7.9670  1.7757  0.0886    0.8713   12.9552    0.0334]; % 2RB 64RX
                       case 1  ; ML=[ 3.4910   51.6973   15.0788   29.4897   1.5461    0.7322   3.3046  12.1719  1.4433  0.1064    0.8491    1.9682    0.0161]; % 1RB 64RX
                    end
                case 8  
                    switch RB_tmp
                        case 16 ; ML=[10.0963  122.6955   23.4735   30.9149    1.3700    0.4195    5.0442    1.9062    0.3000  1.4214    0.6613   21.7805    0.2792]; % 16RB 8RX
                        case 12 ; ML=[ 2.7284  312.1236   20.6815   31.1274    1.5215    0.4229    4.0858    4.7065    0.3205  2.2826    0.6914   28.2467    0.0639]; % 12RB 8RX
                        case 8  ; ML=[40.5728  454.4135   21.1261   15.6235    1.6626    0.4460    3.4245    0.4430    0.2554  2.1937    0.7367   40.0698    0.0289]; %  8RB 8RX
                        case 4  ; ML=[38.7799   94.5508   14.1163   15.5852    1.7164    0.4679   19.1270    2.4472    0.3467  2.9024    0.6932   34.9020    0.0503]; %  4RB 8RX
                        case 2  ; ML=[16.8337  135.7500    9.9632   11.8600    3.0983    0.9391    4.9398    0.3086    0.2378  2.0999    0.5910   15.6284    0.1413]; %  2RB 8RX
                        case 1  ; ML=[ 3.2750    9.3815   20.6265    5.2713    7.0028    0.9057   19.1090   14.9520    1.5004  1.2877    0.5003   16.1563    0.4869]; %  1RB 8RX
                    end
                case 4 
                    switch RB_tmp
                        case 16 ; ML=[49.3937  491.1575   18.1112   28.8541    3.3013    0.9994    2.5731    2.5335   0.1612  4.9003    0.5383   25.2114    0.0808]; %16RB 4RX
                        case 12 ; ML=[ 7.6792  118.3374   18.2318   21.8166    4.8985    0.9869    5.1472    2.7930   0.3350  4.9526    0.5285   48.8772    0.0773]; %12RB 4RX
                        case 8  ; ML=[48.0680  498.8869   10.7923   10.9949    3.1968    0.9964    2.5847    1.2158   0.1992  4.8576    0.4856   42.5081    0.0559]; % 8RB 4RX
                        case 4  ; ML=[26.5473  363.9866    8.8596   14.8423    0.8508    0.9891    8.2441    4.8477   0.3200  7.2645    0.5409   45.8680    0.0577]; % 4RB 4RX
                        case 2  ; ML=[25.3899  189.4908    2.5938   10.0911   10.2593    0.9730    4.6556    7.9189   0.4075  3.9396    0.3846   25.7660    0.0455]; % 2RB 4RX 
                        case 1  ; ML=[ 2.1959   22.8525   19.0413    2.4871    4.2197    0.9532   15.7668    3.4607   1.2732  1.9888    0.2745   18.2319    0.7318]; % 1RB 4RX
                    end

            end
            
            %ML(10:13)=ML_coef;
            
        elseif n==4
            %ML(1:6)=ML_matrix(2,:);
        elseif (n==6) || (n==8)
            %ML(1:6)=ML_matrix(3,:);
        else
            %ML(1:6)=ML_matrix(4,:);
        end
        
        % define current number of symbols for SVD
        N_pilot_svd_cur = min([N_pilot_svd n Nrx]);

        % estimate noise power
        noise_array_reduced=[ data_array(:,n-N_pilot_svd_cur+1:n,upsample_factor*Nfft*(1/4+1/16)+1:upsample_factor*Nfft*(1/2-1/16)) ...
            data_array(:,n-N_pilot_svd_cur+1:n,upsample_factor*Nfft*(3/4+1/16)+1:upsample_factor*Nfft*(1-1/16)) ];
        noise_tmp=squeeze(sum(noise_array_reduced.*conj(noise_array_reduced),2));
        noise_per_sc=sum(squeeze(sum(noise_tmp,1)))/(N_pilot_svd_cur*Nrx*N_used/4);
        
        noise_per_sample1=sum(squeeze(sum(noise_tmp(1:2:Nrx,:),1)))/(N_pilot_svd_cur*upsample_factor*Nfft/4);
        noise_per_sample2=sum(squeeze(sum(noise_tmp(2:2:Nrx,:),1)))/(N_pilot_svd_cur*upsample_factor*Nfft/4);
        noise_per_sample = noise_per_sample1+noise_per_sample2;
        
        
        % initialize array for impulse response reconstruction
        recovered_signal_reduced=zeros(Nrx, 2, upsample_factor*Nfft);
        recovered_TTI=zeros(Nrx, 2*gap-length(pilot_positions), upsample_factor*Nfft);
        
        data_in=zeros(Nrx, 12, upsample_factor*Nfft);
        for k=1:12
            tmp=mean(data_array(:,n-N_pilot_svd_cur+1:n,:),2);
            data_in(:,k,1:N_response)=tmp(:,1,1:N_response);
        end
   
        peak_cancelation=1;
        N_beams=0;
        phi=0;
        
        % peaks compensation process
        for m=1:max_N_peaks

            if m==1
                data_array_reduced=data_array(:,n-N_pilot_svd_cur+1:n,:);
                data_array_new=data_array(:,n-N_pilot_svd_cur+1:n,:);
            else
                data_array_reduced=data_array_new;
            end
            
            
            % find approximate peak position
            samples_power=squeeze(sum(data_array_reduced.*conj(data_array_reduced),2));
            samples_power1=squeeze(sum(samples_power(1:2:Nrx,:),1));
            samples_power2=squeeze(sum(samples_power(2:2:Nrx,:),1));
            samples_power=samples_power1*sub_array_gain+samples_power2;
            
            NS1 = (noise_per_sc*(Nrx+N_pilot_svd_cur)*N_used)/(Nfft*upsample_factor);
            NS2 = (noise_per_sc*Nrx*N_used)/(Nfft*upsample_factor);
            
            
            if m==1
                M_shift=N_shift;
                L=round(ML(1)); % machine learning
                tau=ML(2);     % machine learning
            elseif m==2
                M_shift=N_shift/2;
                L=round(ML(3)); % machine learning 
                tau=ML(4);     % machine learning
            else
                M_shift=0;
            end

            
            a=samples_power(1:N_shift+N_response);
            
            
            b=zeros(1,N_shift+N_response);
            b(1:M_shift)=a(1:M_shift);
            t=L+1:N_response;
            b(N_shift+L+1:N_shift+N_response)=( (t-L-1)/tau ) * noise_per_sample; 
            
            interval=a-b;
            
            width=round(ML(7));
            %width=3;
            if peak_cancelation==0
                 start=max([1 sample_indx-width]);
                 interval(start:sample_indx+width)=zeros(1,sample_indx+width+1-start);
            end
                
            [~,sample_indx]=max(interval);
  
            
            if m==1
                SNR1=(samples_power1(sample_indx)-noise_per_sample1)/noise_per_sample1;
                SNR2=(samples_power2(sample_indx)-noise_per_sample2)/noise_per_sample2;
                SNR=abs(samples_power1(sample_indx)+samples_power2(sample_indx)-noise_per_sample)/noise_per_sample;

                sub_array_gain=( (SNR1+ML(8))/(SNR2+ML(8)) )^(ML(9));
                %sub_array_gain=1;
            end


            
            matrix=squeeze(data_array_reduced(:,:,sample_indx));
            
            N_pilot_svd=2;

            
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    X_vector=squeeze(sum(matrix,2)); % take 64/8/4 amplitudes

    corr_sum1=0;
    corr_sum2=0;

    if 1 > 0
        C=[];
        D=[];
        for sub_array_index=1:2

            A=squeeze(X_vector(sub_array_index:2:end).');
            
            switch Nrx
                
                case 64
                    
                    G=reshape(A,[],8);
                    
                    % angle 1 correlations
                    for i=1:4
                        C=[C conj(G(i,1:7)).*G(i,2:8)];
                    end
                    
                    % angle 2 correlations
                    for j=1:8
                        D=[D (conj(G(1:3,j)).*G(2:4,j)).'];
                    end
                    
                case 8
                    
                    G=reshape(A,[],4);
                    C=conj(G(1,1:3)).*G(1,2:4);
                    
                case 4
                    
                    G=reshape(A,[],2);
                    C=conj(G(1,1:1)).*G(1,2:2);
                    
            end
             

        end
        
        % //////////test antennas correlation////////////
        % cor_vec=(abs(corr_sum1)+abs(corr_sum2))/a(sample_indx);
        
        alpha=angle(sum(C));
        
        switch Nrx
                
            case 64
            
                beta=angle(sum(D));
            
                % make steering vector (the same for each sub-array)
                for k=1:4
                    steering_matrix(k,:)=exp(1i*(alpha*(0:1:7).'+beta*(k-1)));
                end
            
            case 8
                
                steering_matrix(1,:)=exp(1i*(alpha*(0:1:3).'));
                
            case 4
                
                steering_matrix(1,:)=exp(1i*(alpha*(0:1:1).'));       
        end
        

        
        Q=abs(G);
        Q=Q/norm(Q,'fro');
        steering_aligned = steering_matrix.*(Q+ML(12));
        steering_aligned = steering_aligned/norm(steering_aligned,'fro');
        %Sum should be...
        %sum(conj(steering_aligned).*G,'all')
        
        steering_vector=reshape(steering_aligned,1,[]);
        
        U1=steering_vector;

        for sub_array_index=1:2
            
            % extract sub-array signal
            sub_array_matrix=matrix(sub_array_index:2:end,:); % 32 antennas
            
            % calculate projection
            X1=conj(U1)*sub_array_matrix;           
            
            %Correlation test. Should be >> 1/8.
            %cor_test(sub_array_index)=norm(X1)/norm(U1)/norm(sub_array_matrix,'fro');
            
            X1 = X1/(1+ML(10)*NS2/(norm(X1)^2)/2); % 32 antennas
            X1 = U1.'*X1;                             
            X2 = sub_array_matrix - X1;
            X2 = X2*(1 - ML(11)*NS2*(1 - 0.5/Nrx)/(norm(X2, 'fro')^2));
            X_tmp = X1 + X2;
            
            %Power of X1 should be also decreased, but for much smaller
            X(sub_array_index:2:Nrx,1:2)=X_tmp; % 2*32=64 antennas
            
        end
    end
            
%             %Save SVD here if it would be needed
%             if m == 1
%                 
%               % machine learning in future (unsensitive in 1TTI case)
%               
%               [U,S,V] = svd(X2,'econ');
%               SS = S(2,2)^2;
%               NN = 3.023*NS1;                
%               N_pilot_svd = ceil(N_pilot_svd_cur*sqrt(NN/SS) );
%               N_pilot_svd = max(N_pilot_svd, 2);
%               
% %               % doppler shift calculation   
% %               V = V(:,1);
% %               X=U(:,1)*S(1,1)*V';
% %               for k=2:N_pilot_svd_cur
% %                 phase(k-1)=angle(mean(squeeze(X(:,k)).*conj(squeeze(X(:,k-1)))));
% %               end
% %               phi=mean(phase);
% %               
% %               %%%%%%%%%%%%%%%%
% %               phi=0;
% %             
% %               % doppler shift compensation
% %               for k=2:N_pilot_svd_cur
% %                 X2(:,k)=X2(:,k)*exp(-1i*(k-1)*phi);
% %                 data_array_reduced(:,k,:) = data_array_reduced(:,k,:)*exp(-1i*(k-1)*phi);
% %               end
%               
%             end

            % regression model
                clear Q;
                pp=pilot_positions(1) : gap : gap*N_pilot_svd_cur;
                qq=1 : N_pilot_svd_cur*gap;
                    
                    if 2.0022>3.0033   % machine learning in future (insensitive for 1 TTI case)
                        
                        for k=1:Nrx
                            data=squeeze(X(k,:)).';
                            r = corrcoef(pp,data); % Corr coeff is the off-diagonal (1,2) element
                            r = r(1,2);            % Sample regression coefficient
                            xbar = mean(pp);
                            ybar = mean(data);
                            sigx = std(pp);
                            sigy = std(data);
                            slope = r*sigy/sigx;   % Regression line slope
                            Q(k,:) = ybar + slope*(qq - xbar);
                        end
                        
                    else
                        
                        for k=1:Nrx
                            data2=squeeze(X(k,:)).';
                            X(k,:) = ones(N_pilot_svd_cur,1)*mean(data2);
                        end
                        
                        S112 = norm(X, 'fro');
                        
                        %if 1 < 0 %%%%%%%% DUMMY %%%%%%%%%%%
                        if S112^2 < ML(13)*NS2
                          if peak_cancelation==0
                            break;
                          else
                            peak_cancelation=0; % - peak cancelation is OFF 
                          end
                          continue;
                        else
                          peak_cancelation=1; % - peak cancelation is ON 
                          N_beams=N_beams+1;
                        end
                        
                        if peak_cancelation == 1
                          q(N_beams,:)=squeeze(X(:,1))/norm(X(:,1));
                          corr(N_beams)=abs( squeeze(q(N_beams,:))*squeeze(q(1,:))' );
                          position(N_beams)=sample_indx;
                        end 
                       
                        
%                         if 1<0
%                             % reduce noise over beams correlation
%                             if corr(N_beams)>ML(6) && N_beams>1
%                                 U1 = conj(squeeze(q(1,:)));
%                                 X01 = U1*X;
%                                 X02 = X - (U1'*X01);
%                                 %Put ML here?
%                                 X01 = X01*(1 - ML(5)*NS2/(Nrx*norm(X01)^2));
%                                 X01 = U1'*X01;
%                                 %Put ML here?
%                                 X02 = X02*(1 - 1*NS2*(1 - 1.0/Nrx)/(norm(X02, 'fro')^2));
%                                 X = X01 + X02;
%                             else
%                                 %Put ML here?
%                                 X = X*(1 -1*NS2/S112^2);
%                             end
%                         end
                        
                        for k=1:Nrx
                            data2=squeeze(X(k,:)).';
                            Q(k,:) = ones(N_pilot_svd_cur*gap,1)*mean(data2);
                        end

                    end
                    

            theta=phi/gap;
            % doppler shift reconstruction in TTI
            for k=pilot_positions(1)+1 : gap*N_pilot_svd_cur
                Q(:,k)=Q(:,k)*exp(1i*(k-pilot_positions(1))*theta);
            end
            for k=pilot_positions(1)-1 : -1 :1
                Q(:,k)=Q(:,k)*exp(1i*(k-pilot_positions(1))*theta);
            end
            
            % extract pilots with history (several TTIs)
            X=Q(:,pilot_positions(1) : gap : gap*N_pilot_svd_cur);
            
            % extract current data (1 TTI)
            T=Q(:,(N_pilot_svd_cur-2)*gap+1 : N_pilot_svd_cur*gap);
            H_data(:,data_positions1-0)=T(:,data_positions1);
            H_data(:,data_positions2-1)=T(:,data_positions2);
            H_data(:,data_positions3-2)=T(:,data_positions3);

           % recover time domain pilot channel 
           tmp=circshift(SINC_t,[0 sample_indx-1]);
           X_vector=reshape(X,1,[]);
           temp=X_vector.'*tmp;
           SINC_peak=peak_cancelation*reshape(temp,Nrx,N_pilot_svd_cur,[]);
           recovered_signal_reduced=recovered_signal_reduced+squeeze(SINC_peak(:,N_pilot_svd_cur-1:N_pilot_svd_cur,:));
           data_array_new=data_array_reduced-SINC_peak;
           
           
           % recover beam domain channel
           if peak_cancelation == 1
               delay=-(sample_indx-1-N_shift);
               subcarriers=(1:N_used)-1-(N_used)/2;
               beam_amplitudes(N_beams,1:N_used)=(norm(X(:,1))/SINC_scale)*exp(2*pi*1i*subcarriers*delay/(upsample_factor*Nfft));
               beam_transform_matrix(:,N_beams)=X(:,1)/norm(X(:,1));
           end

           % save channel for current TTI on data positions
           H_data_vector=reshape(H_data,1,[]);
           temp=H_data_vector.'*tmp;
           SINC_peak=peak_cancelation*reshape(temp,Nrx,2*gap-length(pilot_positions),[]);
           recovered_TTI=recovered_TTI+SINC_peak;

        end
        
        % channel in pilots positions
        recovered_pilots(:,n-1:n,:)=recovered_signal_reduced;

        
        if N_beams==0
            SNR=0;
            recovered_TTI=data_in/100;
        end

        
    end
    
   

h_srs_f=beam_transform_matrix*beam_amplitudes;


N_ports=estimationParams.N_ports;
% generate DFT-based beams matrix
W_pol = dftmtx(2);
W_hor = dftmtx(8); %.*repmat(1i*(1:8)*pi/8,8,1);
W_vert = dftmtx(4);
Wrez = kron(W_hor, W_vert);
Wfin = kron(Wrez, W_pol);
D = abs(diag(Wfin'*Wfin));
Wfin = Wfin * diag(1.0./sqrt(D));

%Wfin  = (dftmtx(Nrx));
D = abs(diag(Wfin'*Wfin));
Wfin = Wfin * diag(1.0./sqrt(D));

beam_power = sum( abs(h_srs_f.'*conj(Wfin)).^2 , 1);
[~, beam_indx] = sort(beam_power, 'descend');

% % Print 64 beams power
% beam_power(beam_indx(1:64))

beam_matrix=Wfin(:,beam_indx(1:N_ports));

% [Qbtm, Rbtm] = qr(beam_transform_matrix,0);
% beam_amplitudes = Rbtm*beam_amplitudes;
% beam_matrix = Qbtm;


  