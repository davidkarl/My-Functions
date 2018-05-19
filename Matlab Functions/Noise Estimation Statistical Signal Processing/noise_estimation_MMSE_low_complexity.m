function [enhanced_speech, noise_psd_matrix,T]=noise_estimation_MMSE_low_complexity(input_signal,Fs)
%%%% Output parameters:  noisePowMat:  matrix with estimated noise PSD for each frame
%%%%                    shat:      estimated clean signal
%%%%                    T:      processing time


%Initialisation of GGD parameters:
MIN_GAIN = eps;
gamma = 1;
nu = 0.6;

%Get matrices with the gain functions for the different estimators needed
%within a range of apriori and aposteriori SNR:
[g_dft,g_mag,g_mag2] = Tabulate_gain_functions(gamma,nu);

%Initialize audio parameters:
alpha_apriori_SNR_smoothing_factor_DD = 0.98;
apriori_SNR_lower_limit = eps;
Fs_reference = 8000;
samples_per_frame_reference = 256;
samples_per_frame = samples_per_frame_reference*Fs/Fs_reference;
non_overlapping_samples_per_frame = samples_per_frame/2; 
number_of_frames =  floor((length(input_signal) - 2*samples_per_frame)/non_overlapping_samples_per_frame );
analysis_window  = sqrt(hanning(samples_per_frame ));
synthesis_window = sqrt(hanning(samples_per_frame ));
clear_frame_estimate_fft = [];
fft_size = samples_per_frame;
enhanced_speech = zeros(size(input_signal));

%Initialize noise tracker (ASSUMES THE FIRST 5 FRAMES ARE NOISE ONLY):
for I=1:5
    noisy_frame=analysis_window .* input_signal((I-1)*non_overlapping_samples_per_frame+1:(I-1)*non_overlapping_samples_per_frame+samples_per_frame);
    noisy_dft_frame_matrix(:,I)=fft(noisy_frame,fft_size);
end
noise_psd_previous_smoothed = mean(abs(noisy_dft_frame_matrix(1:samples_per_frame/2+1,1:end)).^2,2);

%Initialize minimum tracking mat, noise psd mat, and gamma factor:
min_mat = zeros(fft_size/2+1,floor(0.8*Fs/non_overlapping_samples_per_frame));
noise_psd_matrix = zeros(samples_per_frame/2+1,number_of_frames);
Rprior_dB = -40:1:100;
Rprior_linear = 10.^(Rprior_dB(:)/10);
tabel_inc_gamma = gammainc(1./(1+Rprior_linear),2);
noise_psd_smoothing_factor = 0.8;

%%%%%%%%%%%%%%Algorithm:
tic
for frame_counter=1:number_of_frames
    
    %get current frame:
    current_frame_indices = (frame_counter-1)*non_overlapping_samples_per_frame+1:(frame_counter-1)*non_overlapping_samples_per_frame+samples_per_frame;
    noisy_frame = analysis_window .* input_signal(current_frame_indices);
    noisey_frame_fft = fft(noisy_frame,samples_per_frame);
    noisey_frame_fft = noisey_frame_fft(1:samples_per_frame/2+1);
    noisy_frame_ps_current = abs(noisey_frame_fft).^2;
    clean_frame_estimate_ps = abs(clear_frame_estimate_fft).^2;
    
    %estimate aposteriori and apriori SNR before bias:
     a_post_snr_before_bias_compensation =  noisy_frame_ps_current./(noise_psd_previous_smoothed(1:fft_size/2+1));
     if frame_counter==1,
         a_priori_snr_using_previous_noise = max( a_post_snr_before_bias_compensation-1 , apriori_SNR_lower_limit);
     else
         a_priori_snr_using_previous_noise = max( alpha_apriori_SNR_smoothing_factor_DD*( clean_frame_estimate_ps(1:fft_size/2+1))./( noise_psd_previous_smoothed(1:fft_size/2+1)) ...
             + (1-alpha_apriori_SNR_smoothing_factor_DD)*(a_post_snr_before_bias_compensation-1),apriori_SNR_lower_limit);
     end
    speech_psd_previous_noise_and_apriori_without_bias_compensation = ...
        a_priori_snr_using_previous_noise .* noise_psd_previous_smoothed;
    
    
    %estimate noise psd and apply bias compensation:
    a_post_snr = (noisy_frame_ps_current)./noise_psd_previous_smoothed;
    xiest=max(a_post_snr-1,apriori_SNR_lower_limit);
    gain_function=(xiest./((xiest+1).*a_post_snr)).*(1+a_post_snr./(xiest.*(xiest+1)));
    [gains]=lookup_inc_gamma_in_table(tabel_inc_gamma ,speech_psd_previous_noise_and_apriori_without_bias_compensation./noise_psd_previous_smoothed,-40:1:100,1);
    total_estimated_psd = noise_psd_previous_smoothed + speech_psd_previous_noise_and_apriori_without_bias_compensation;
    bias_factor_for_current_noise_estimate = noise_psd_previous_smoothed ./ ( (gains).*total_estimated_psd + noise_psd_previous_smoothed.*exp(-noise_psd_previous_smoothed./total_estimated_psd) );
    current_noise_estimate = bias_factor_for_current_noise_estimate.*gain_function.*(noisy_frame_ps_current);
    noise_psd_current_smoothed = noise_psd_smoothing_factor*noise_psd_previous_smoothed ...
        + (1-noise_psd_smoothing_factor)*current_noise_estimate;
    
    
    %append to mat:
    min_mat = [min_mat(:,end-floor(0.8*Fs/non_overlapping_samples_per_frame)+2:end),noisy_frame_ps_current(1:fft_size/2+1)    ];
    noise_psd_current_smoothed = max(noise_psd_current_smoothed,min(min_mat,[],2));
    
    %update apriori and aposteriori SNRs with NEW noise psd:
    a_post_snr = noisy_frame_ps_current./(noise_psd_current_smoothed(1:fft_size/2+1));
    if frame_counter==1,
        a_priori_snr_using_current_noise = max( a_post_snr-1,apriori_SNR_lower_limit);
    else
        a_priori_snr_using_current_noise = max(alpha_apriori_SNR_smoothing_factor_DD*(clean_frame_estimate_ps(1:fft_size/2+1))./(noise_psd_current_smoothed(1:fft_size/2+1)) + (1-alpha_apriori_SNR_smoothing_factor_DD)*(a_post_snr-1),apriori_SNR_lower_limit);
    end


    
    %find gain for current apriori and aposteriori SNR:
    [gain] = lookup_gain_in_table(g_mag,a_post_snr,a_priori_snr_using_current_noise,-40:1:50,-40:1:50,1);
    gain = max(gain,MIN_GAIN);
    noise_psd_matrix(:,frame_counter) = noise_psd_current_smoothed;
    clear_frame_estimate_fft = gain.*noisey_frame_fft(1:fft_size/2+1);
    enhanced_speech(current_frame_indices) = enhanced_speech(current_frame_indices) ...
        + synthesis_window.*real(ifft( [clear_frame_estimate_fft;flipud(conj(clear_frame_estimate_fft(2:end-1)))]));
    
    noise_psd_previous_smoothed = noise_psd_current_smoothed;
end
T=toc;







function [gains]=lookup_inc_gamma_in_table(G,a_priori,a_priori_range,step)
    a_prioridb=round(10*log10(a_priori)/step)*step;
    [Ia_priori]=min(max(min(a_priori_range),a_prioridb), max(a_priori_range));
    Ia_priori=Ia_priori-min(a_priori_range)+1;
    Ia_priori=Ia_priori/step;
    gains=G(Ia_priori);
end



function [a_post_snr,a_priori_snr]=estimate_snrs(noisy_frame_ps,fft_size, noise_psd_current ,apriori_SNR_lower_limit, alpha_apriori_SNR_smoothing_factor_DD, frame_counter, clean_frame_estimate_ps)  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%This m-file estimates the a priori SNR 
%%%%
%%%%Input parameters:   noisy_dft_frame:    noisy DFT frame
%%%%                    fft_size:           fft size 
%%%%                    noise_psd:          estimated noise PSD of previous frame, 
%%%%                    SNR_LOW_LIM:        Lower limit of the a priori SNR
%%%%                    ALPHA:              smoothing parameter in dd approach ,
%%%%                    I:                  frame number 
%%%%                    clean_est_dft_frame:estimated clean frame of frame
%%%%                                         I-1
%%%%                   
%%%%Output parameters:  a_post_snr:    a posteriori SNR 
%%%%                                      
%%%%                    a_priori_snr: estimated a priori SNR
%%%%                  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%Author: Richard C. Hendriks, 15/4/2010
%%%%%%%%%%%%%%%%%%%%%%%copyright: Delft university of Technology
%%%%%%%%%%%%%%%%%%%%% 

a_post_snr = noisy_frame_ps./(noise_psd_current(1:fft_size/2+1));%a posteriori SNR
if frame_counter==1,
    a_priori_snr=max( a_post_snr-1,apriori_SNR_lower_limit);
else
    a_priori_snr=max(alpha_apriori_SNR_smoothing_factor_DD*( clean_frame_estimate_ps(1:fft_size/2+1))./(   noise_psd_current(1:fft_size/2+1))+(1-alpha_apriori_SNR_smoothing_factor_DD)*(a_post_snr-1),apriori_SNR_lower_limit);
end

end


function [gains]=lookup_gain_in_table(G,a_post,a_priori,a_post_range,a_priori_range,step);
% function [gains]=lookup_gain_in_table(G,a_post,a_priori,a_post_range,a_priori_range,step);
% This function selects the right gain value from the table G, given
% vectors with a priori and a posteriori SNRs
%
% INPUT variables:
% G: Matrix with gain values for speech DFT or magnitude estimation,
% evaluated at all combinations of a priori and a posteriori SNR in the
% input variables Rksi and Rgam. 
% 
%
% a_priori: Array of "a priori" SNR (SNRprior) values for which values
% have to be selected from the gain table   NOTE: The values must be in dBs.
% a_post: Array of "a posteriori" SNR (SNRpost) values for which values
% have to be selected from the gain table  NOTE: The values must be in dBs.
%
% a_post_range: The range of "a posteriori" SNR values
%
% a_priori_range: The range of "a priori" SNR values
%
% step: step is the stepsize in db's in the table
%
% OUTPUT variables:
% gains: Matrix with gain values that are selected from the gain table G
% 
%
% Copyright 2007: Delft University of Technology, Information and
% Communication Theory Group. The software is free for non-commercial use.
% This program comes WITHOUT ANY WARRANTY.
%
% Last modified: 22-11-2007.




a_prioridb=round(10*log10(a_priori)/step)*step;
a_postdb=round(10*log10(a_post)/step)*step;
[Ia_post]=min(max(min(a_post_range),a_postdb), max(a_post_range));
Ia_post=Ia_post-min(a_post_range)+1;
Ia_post=Ia_post/step;
[Ia_priori]=min(max(min(a_priori_range),a_prioridb), max(a_priori_range));
Ia_priori=Ia_priori-min(a_priori_range)+1;
Ia_priori=Ia_priori/step;

gains=G(Ia_priori+(Ia_post-1)*length(G(:,1))); 
end







    
    
