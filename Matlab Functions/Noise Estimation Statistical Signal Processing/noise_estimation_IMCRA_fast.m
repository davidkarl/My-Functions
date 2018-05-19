function parameters = noise_estimation_IMCRA_fast(current_frame_ps,parameters)
%         parameters = struct('n',2,'len',len_val,'noise_cap',ns_ps,'noise_tild',ns_ps,'gamma',ones(len_val,1),'Sf',Sf_val,...
%             'Smin',Sf_val,'S',Sf_val,'S_tild',Sf_val,'GH1',ones(len_val,1),'Smin_tild',Sf_val,'Smin_sw',Sf_val,'Smin_sw_tild',Sf_val,...
%             'stored_min',max(ns_ps)*ones(len_val,U_val),'stored_min_tild',max(ns_ps)*ones(len_val,U_val)','u1',1,'u2',1,'j',2,...
%             'alpha_d',0.85,'alpha_s',0.9,'U',8,'V',15,'Bmin',1.66,'gamma0',4.6,'gamma1',3,'psi0',1.67,'alpha',0.92,'beta',1.47,...
%             'b',b_val,'Sf_tild',Sf_tild_val);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GET PARAMETERS FROM PREVIOUS SESSION:
tic
%frame size:
current_frame_size = parameters.len;

%counters:
frame_counter = parameters.n;
spectrum_minimum_buildup_counter = parameters.j;
number_of_spectral_minimum_buildup_blocks_to_remember = parameters.U;
number_of_frames_to_buildup_spectrum_minimum = parameters.V;
u1 = parameters.u1;
u2 = parameters.u2;

%power spectrums:
raw_window_averaged_ps = parameters.Sf;
raw_window_averaged_ps_smoothed_min = parameters.Smin;
raw_window_averaged_ps_smoothed = parameters.S;
raw_window_averaged_primarily_noise_ps_smoothed = parameters.S_tild;
raw_window_averaged_primarily_noise_ps = parameters.Sf_tild;
raw_window_averaged_primarily_noise_ps_smoothed_min = parameters.Smin_tild;
raw_window_averaged_ps_smoothed_min_sw = parameters.Smin_sw;
raw_window_averaged_primarily_noise_ps_smoothed_min_sw = parameters.Smin_sw_tild;

%Smoothing factors:
apriori_SNR_smoothing_factor = parameters.alpha;
raw_window_averaged_smoothed_ps_smoothing_factor = parameters.alpha_s;
final_noise_smoothing_factor_smoothing_factor = parameters.alpha_d;

%PS to PS minimum ratio SNR thresholds:
current_ps_to_smoothed_ps_minimum_aposteriori_SNR_threshold = parameters.gamma0;
smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_threshold = parameters.psi0;

%Speech presence probability threshold:
current_noise_ps_to_smoothed_noise_ps_minimum_upper_threshold = parameters.gamma1;

%Spectrum smoothing window:
small_3_term_window_spectrum_smoother_buffered = parameters.b;

%Get previous session spectrum and SNR parameters:
raw_ps_to_smoothed_noise_ps_aposteriori_SNR_previous = parameters.gamma;
final_noise_ps_after_beta_correction_previous = parameters.noise_ps;
final_noise_ps_before_beta_correction_previous = parameters.noise_tild;

%Previous (log-MMSE) gain function (GH1):
spectral_magnitude_gain_function_previous = parameters.GH1;

%Minimum to nominal ps correction factor:
minimum_ps_correction_factor = parameters.Bmin;

%Final noise correction factor (beta):
final_noise_correction_factor = parameters.beta;

%stored specral minimas:
stored_raw_window_averaged_ps_smoothed_min = parameters.stored_min;
stored_noise_window_averaged_ps_smoothed_min = parameters.stored_min_tild;
disp('get initial paramteres:');
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%UPDATE AND CALCULATE THINGS INVOLVING RAW (SIGNAL ONLY) POWER SPECTRUM:
%Calculate current raw ps to smoothed noise ps aposteriori SNR (gamma): 
tic
raw_ps_to_smoothed_noise_ps_aposteriori_SNR_current = current_frame_ps ./ final_noise_ps_after_beta_correction_previous;
disp('divide current frame ps by previous final estimated noise ps:');
toc

%Calculate apriori SNR estimate (epsilon):
tic
previous_smoothed_apriori_SNR_estimate = spectral_magnitude_gain_function_previous.^2 .* raw_ps_to_smoothed_noise_ps_aposteriori_SNR_previous;
disp('multiply previous raw ps to smoothed noise ps ratio by gain function squared for previous apriori SNR estimate:');
toc

tic
current_raw_apriori_SNR_estimate_using_aposteriori_SNR_estimate = max(raw_ps_to_smoothed_noise_ps_aposteriori_SNR_current-1,0); 
disp('estimate current apriori SNR using aposteriori SNR estimate max(bla-1,0):');
toc

tic
apriori_SNR_smoothed_estimate = apriori_SNR_smoothing_factor * previous_smoothed_apriori_SNR_estimate ...
    + (1-apriori_SNR_smoothing_factor) * current_raw_apriori_SNR_estimate_using_aposteriori_SNR_estimate;
disp('smooth apriori SNR estimate:');
toc

%Compute current v_k and spectral magnitude gain function:
tic
v_k = (apriori_SNR_smoothed_estimate./(1+apriori_SNR_smoothed_estimate)) .* raw_ps_to_smoothed_noise_ps_aposteriori_SNR_current;
disp('compute current v_k with apriori and aposteriori SNR estimates:');
toc

tic
spectral_magnitude_gain_function_current = (apriori_SNR_smoothed_estimate./(1+apriori_SNR_smoothed_estimate)) .*exp(1/2*expint(v_k));
disp('calculate current spectral magnitude gain function by log-MMSE (expint(v_k)):');
toc

%Calculate window-averaged RAW power spectrum (SIGNAL+NOISE):
tic
power_spectrum_buffered = buffer(current_frame_ps,3,2);
small_3_term_window_spectrum_smoother = small_3_term_window_spectrum_smoother_buffered(:,1);
raw_window_averaged_ps = power_spectrum_buffered'*small_3_term_window_spectrum_smoother_buffered(:,1);
raw_window_averaged_ps(1)=current_frame_ps(1);
raw_window_averaged_ps(end)=current_frame_ps(end);
disp('smooth current raw ps using b smoothing vector (using loop not conv(!!!)):');
toc

tic
raw_window_averaged_ps_smoothed = raw_window_averaged_smoothed_ps_smoothing_factor*raw_window_averaged_ps_smoothed ... 
                                + (1-raw_window_averaged_smoothed_ps_smoothing_factor)*raw_window_averaged_ps;
disp('smooth raw window ps:');
toc
   
tic
raw_window_averaged_ps_smoothed_min = min(raw_window_averaged_ps_smoothed_min,raw_window_averaged_ps_smoothed);
raw_window_averaged_ps_smoothed_min_sw = min(raw_window_averaged_ps_smoothed_min_sw,raw_window_averaged_ps_smoothed);
disp('find min of raw, window convolved, smoothed ps (twice):');
toc

%Calculate ps to smoothed Pmin aposteriori SNR (gamma_min) and smooth ps to smooth ps minimum (psi_min):
tic
current_ps_to_smoothed_ps_minimum_aposteriori_SNR_gamma_min = current_frame_ps ./ (minimum_ps_correction_factor*raw_window_averaged_ps_smoothed_min);
smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_psi_min = raw_window_averaged_ps_smoothed ./ (minimum_ps_correction_factor*raw_window_averaged_ps_smoothed_min);
disp('get current raw ps and smoothed ps to smoothed ps minimum aposteriori SNR (division with beta correction:');
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%UPDATE AND CALCULATE THINGS INVOLVING PRIMARILY NOISE POWER SPECTRUM:
%Build logical mask which has ones where speech is deemed absent (and we might need to update ps around it):
tic
logical_mask_where_speech_is_absent = zeros(current_frame_size,1);
logical_mask_where_speech_is_absent(current_ps_to_smoothed_ps_minimum_aposteriori_SNR_gamma_min<current_ps_to_smoothed_ps_minimum_aposteriori_SNR_threshold ... 
    & smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_psi_min<smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_threshold) = 1;  
disp('update logical mask indices to 1 where find condition holds:');
toc
 
%Update Sf_tilde to update S_tilde in frequencies where in their vicinity were found to contain only noise:
tic
ones_mat = ones(1,3);
logical_mask_buffered = buffer(logical_mask_where_speech_is_absent,3,2);
logical_mask_where_speech_is_absent_including_neighbour_search = (ones_mat*logical_mask_buffered==0);
denom_buffered = logical_mask_buffered.*small_3_term_window_spectrum_smoother_buffered;
raw_window_averaged_primarily_noise_ps = sum(denom_buffered.*power_spectrum_buffered)./sum(denom_buffered);
raw_window_averaged_primarily_noise_ps = raw_window_averaged_primarily_noise_ps';
%Sample 1: 
if logical_mask_where_speech_is_absent(1)==0
    raw_window_averaged_primarily_noise_ps(1) = raw_window_averaged_primarily_noise_ps_smoothed(1);
else
    raw_window_averaged_primarily_noise_ps(1) = current_frame_ps(1);
end
%Sample 2:
raw_window_averaged_primarily_noise_ps(2) = sum(small_3_term_window_spectrum_smoother.*[logical_mask_where_speech_is_absent(1); logical_mask_where_speech_is_absent(2); logical_mask_where_speech_is_absent(3)].*[current_frame_ps(1); current_frame_ps(2); current_frame_ps(3)]) ...
    / sum(small_3_term_window_spectrum_smoother.*[logical_mask_where_speech_is_absent(1); logical_mask_where_speech_is_absent(1); logical_mask_where_speech_is_absent(3)]);
%Sample N-1:
raw_window_averaged_primarily_noise_ps(end-1) = sum(small_3_term_window_spectrum_smoother.*[logical_mask_where_speech_is_absent(end-2); logical_mask_where_speech_is_absent(end-1); logical_mask_where_speech_is_absent(end)].*[current_frame_ps(end-2); current_frame_ps(end-1); current_frame_ps(end)]) ...
    / sum(small_3_term_window_spectrum_smoother.*[logical_mask_where_speech_is_absent(end-2); logical_mask_where_speech_is_absent(end-1); logical_mask_where_speech_is_absent(end)]);
%Sample N:
if logical_mask_where_speech_is_absent(end)==0        
    raw_window_averaged_primarily_noise_ps(end) = raw_window_averaged_primarily_noise_ps_smoothed(end);
else
    raw_window_averaged_primarily_noise_ps(end) = current_frame_ps(end);
end
%Deal with samples where speech is total absent and denominator is zero:
raw_window_averaged_primarily_noise_ps(logical_mask_where_speech_is_absent_including_neighbour_search) = raw_window_averaged_primarily_noise_ps_smoothed(logical_mask_where_speech_is_absent_including_neighbour_search);
disp('average using small smoother window raw ps samples that contain primarily noise using indicators (using loop not conv!!!):');
toc

%Calculate raw window averaged PRIMARILY NOISE ps smoothed:
tic
raw_window_averaged_primarily_noise_ps_smoothed = raw_window_averaged_smoothed_ps_smoothing_factor*raw_window_averaged_primarily_noise_ps_smoothed ...
                                                + (1-raw_window_averaged_smoothed_ps_smoothing_factor)*raw_window_averaged_primarily_noise_ps;
disp('smooth raw window averaged primarily noise ps:');
toc
   
tic
raw_window_averaged_primarily_noise_ps_smoothed_min = min(raw_window_averaged_primarily_noise_ps_smoothed_min,raw_window_averaged_primarily_noise_ps_smoothed);
raw_window_averaged_primarily_noise_ps_smoothed_min_sw = min(raw_window_averaged_primarily_noise_ps_smoothed_min_sw,raw_window_averaged_primarily_noise_ps_smoothed);
disp('find min of raw window averaged primarily noise ps (twice):');
toc

%Calculate noise ps to smoothed P_noise_min aposteriori SNR (gamma_min_tilde) 
%and smooth noise ps to smooth noise ps minimum (psi_min_tilde):
tic
current_noise_ps_to_smoothed_noise_ps_minimum_aposteriori_SNR = current_frame_ps ./ (minimum_ps_correction_factor*raw_window_averaged_primarily_noise_ps_smoothed_min);
smoothed_ps_to_smoothed_noise_ps_minimum_aposteriori_SNR = raw_window_averaged_ps_smoothed ./ (minimum_ps_correction_factor*raw_window_averaged_primarily_noise_ps_smoothed_min);
disp('calculate raw ps to smoothed min noise ps after beta correction aposteriori SNR and another one like that:');
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%v%%%%%%%%%%%%%%%%%%%%
%CALCULATE APPROXIMATE A-PRIORI SPEECH ABSENCE PROBABILITY AND CONDITIONAL SPEECH PRESENCE PROBABILITY:
q_apriori_speech_absence_probability = zeros(current_frame_size,1);
tic
indices_where_speech_is_deemed_absolutely_absent = find(current_noise_ps_to_smoothed_noise_ps_minimum_aposteriori_SNR<=1 & smoothed_ps_to_smoothed_noise_ps_minimum_aposteriori_SNR<smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_threshold);
disp('find indices where speech is deemed absolutely absent:');
toc

tic
indices_where_speech_is_deemed_maybe_present = setdiff([1:current_frame_size],indices_where_speech_is_deemed_absolutely_absent);
disp('find indices where speech is NOT deemed absolutely absent using setdiff:');
toc

tic
if (~isempty(indices_where_speech_is_deemed_absolutely_absent))
    q_apriori_speech_absence_probability(indices_where_speech_is_deemed_absolutely_absent) = 1;
end
disp('put 1 into apriori speech absence probability in indices which were found to not absolutely not contain speech:');
tic

tic
indices_where_speech_is_deemed_probably_absent = find(current_noise_ps_to_smoothed_noise_ps_minimum_aposteriori_SNR>1 & current_noise_ps_to_smoothed_noise_ps_minimum_aposteriori_SNR<current_noise_ps_to_smoothed_noise_ps_minimum_upper_threshold & smoothed_ps_to_smoothed_noise_ps_minimum_aposteriori_SNR<smoothed_ps_to_smoothed_ps_minimum_aposteriori_SNR_threshold);
disp('find indices where speech is deemed probably absent:');
toc

tic
if (~isempty(indices_where_speech_is_deemed_probably_absent))
    q_apriori_speech_absence_probability(indices_where_speech_is_deemed_probably_absent)=(current_noise_ps_to_smoothed_noise_ps_minimum_upper_threshold-current_noise_ps_to_smoothed_noise_ps_minimum_aposteriori_SNR(indices_where_speech_is_deemed_probably_absent))/(current_noise_ps_to_smoothed_noise_ps_minimum_upper_threshold-1);             
end
disp('put some calculated value into indices where speech was deemed probably absent:');
toc

%Get final speech presence probability:
%p=1./(1+((q./(1-q)).*(1+eps_cap).*exp(-v)));
tic
p_speech_presence_probability = zeros(current_frame_size,1);
if (~isempty(indices_where_speech_is_deemed_maybe_present))
    temp1 = q_apriori_speech_absence_probability(indices_where_speech_is_deemed_maybe_present)./(1-q_apriori_speech_absence_probability(indices_where_speech_is_deemed_maybe_present));
    temp2 = 1 + apriori_SNR_smoothed_estimate(indices_where_speech_is_deemed_maybe_present);
    temp3 = exp(-v_k(indices_where_speech_is_deemed_maybe_present));
    p_speech_presence_probability(indices_where_speech_is_deemed_maybe_present) = (1 + temp1.*temp2.*temp3).^-1;
end
disp('calcuate final speech present probability using formula:');
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CALCULATE FINAL NOISE PS USING SPEECH PRESENCE PROBABILITY:
tic
final_noise_ps_smoothing_factor = final_noise_smoothing_factor_smoothing_factor ...
 + (1-final_noise_smoothing_factor_smoothing_factor)*p_speech_presence_probability;
disp('calculate final noise ps smoothing FACTOR:');
toc

tic
final_noise_ps_before_beta_correction_current = final_noise_ps_smoothing_factor.*final_noise_ps_before_beta_correction_previous ...
                                                + (1-final_noise_ps_smoothing_factor).*current_frame_ps; 
disp('smooth final noise ps before beta correction:');
toc
             
tic
final_noise_ps_current = final_noise_correction_factor * final_noise_ps_before_beta_correction_current;
disp('get final noise ps using beta correction:');
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%REFRESH AND UPDATE SPECTRAL MINIMUMS:
tic
spectrum_minimum_buildup_counter = spectrum_minimum_buildup_counter+1;
if spectrum_minimum_buildup_counter==number_of_frames_to_buildup_spectrum_minimum
   
    stored_raw_window_averaged_ps_smoothed_min(:,u1) = raw_window_averaged_ps_smoothed_min_sw;
    u1 = u1+1;
    if u1==number_of_spectral_minimum_buildup_blocks_to_remember+1; 
       u1 = 1;
    end
    raw_window_averaged_ps_smoothed_min = min(stored_raw_window_averaged_ps_smoothed_min,[],2);
    raw_window_averaged_ps_smoothed_min_sw = raw_window_averaged_ps_smoothed;
    
    stored_noise_window_averaged_ps_smoothed_min(:,u2) = raw_window_averaged_primarily_noise_ps_smoothed_min_sw;
    u2 = u2+1;
    if u2==number_of_spectral_minimum_buildup_blocks_to_remember+1; 
       u2 = 1;
    end
    raw_window_averaged_primarily_noise_ps_smoothed_min = min(stored_noise_window_averaged_ps_smoothed_min,[],2);
    raw_window_averaged_primarily_noise_ps_smoothed_min_sw = raw_window_averaged_primarily_noise_ps_smoothed;
    
    spectrum_minimum_buildup_counter = 0;
end
disp('refresh and update spectral minimums:');
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
%UPDATE PARAMETERS FOR NEXT ROUND:
parameters.n = frame_counter+1;
parameters.gamma = raw_ps_to_smoothed_noise_ps_aposteriori_SNR_current;
parameters.GH1 = spectral_magnitude_gain_function_current;
parameters.Sf = raw_window_averaged_ps;
parameters.Smin = raw_window_averaged_ps_smoothed_min;
parameters.S = raw_window_averaged_ps_smoothed;
parameters.S_tild = raw_window_averaged_primarily_noise_ps_smoothed;
parameters.Smin_tild = raw_window_averaged_primarily_noise_ps_smoothed_min;
parameters.Smin_sw = raw_window_averaged_ps_smoothed_min_sw;
parameters.Smin_sw_tild = raw_window_averaged_primarily_noise_ps_smoothed_min_sw;
parameters.stored_min = stored_raw_window_averaged_ps_smoothed_min;
parameters.stored_min_tild = stored_noise_window_averaged_ps_smoothed_min;
parameters.u1 = u1;
parameters.u2 = u2;
parameters.j = spectrum_minimum_buildup_counter;
parameters.noise_tild=final_noise_ps_before_beta_correction_current;
parameters.noise_ps = final_noise_ps_current;
disp('update parameters for next round:');
toc
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');