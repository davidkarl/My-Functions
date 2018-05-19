function parameters = noise_estimation_connected_time_frequency_regions(current_frame_ps,parameters)
% parameters = struct('n',2,'len',len_val,'ad',0.95,'as',0.8,'ap',0.2,'beta',0.8,'beta1',0.98,'gamma',0.998,'alpha',0.7,...
%             'pk',zeros(len_val,1),'noise_ps',ns_ps,'pxk_old',ns_ps,'pxk',ns_ps,'pnk_old',ns_ps,'pnk',ns_ps);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Get Initial Parameters:

%Counters:
window_half_length = parameters.D;
current_frame_size = parameters.len;
u1 = parameters.u1;
j = parameters.j;
spectral_smoothing_window = parameters.b;
number_of_frames_to_buildup_spectrum_minimum = parameters.V;
number_of_spectral_minimum_buildup_blocks_to_remember = parameters.U;

%Speech presence thresholds:
gamma1 = parameters.gamma1;
gamma2 = parameters.gamma2;

%Smoothing factors parameters:
alpha_max = parameters.alpha_max;
R_total_power_to_min_total_power_smoothing_factor = parameters.beta_min;

%noise Correction factors:
alpha_c_noise_ps_smoother_correction_factor_smoothed = parameters.alpha_c;

%Power spectrums:
noise_ps_smoothed_previous = parameters.SmthdP;
noise_ps_final_smoothed = parameters.noise_ps;

%Spectrum minimums:
Rmin_old = parameters.Rmin_old;
Pmin = parameters.Pmin;
Pmin_sw = parameters.Pmin_sw;
stored_min = parameters.stored_min;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%Spectrally smooth raw frame ps:
current_frame_ps_spectrally_smoothed = smoothing(current_frame_ps, spectral_smoothing_window, window_half_length);

%Calculate alpha_c correction factor for alpha_opt noise smoothing factor:
R_averaged_former_to_corrent_periodogram_total_power_ratio = sum(noise_ps_smoothed_previous)/sum(current_frame_ps);
alpha_c_noise_ps_smoother_correction_factor_current = 1/(1+(R_averaged_former_to_corrent_periodogram_total_power_ratio-1)^2);

%Smooth alpha_c correction factor and ground it to a minimum of 0.3:
alpha_c_noise_ps_smoother_correction_factor_smoothed = alpha_c_noise_ps_smoother_correction_factor_smoothed*0.7 ...
         + 0.3*max(alpha_c_noise_ps_smoother_correction_factor_current, 0.7);
alpha_opt_noise_ps_smoothing_factor = (alpha_max*alpha_c_noise_ps_smoother_correction_factor_smoothed)...
         ./ (1 + (noise_ps_smoothed_previous./(noise_ps_final_smoothed+eps)-1).^2);

%Calculate current smoothed noise ps:
noise_ps_smoothed_current = alpha_opt_noise_ps_smoothing_factor.*noise_ps_smoothed_previous ...
                          + (1-alpha_opt_noise_ps_smoothing_factor).*current_frame_ps_spectrally_smoothed;

%Temporal smoothing
P_min_previous_sum = sum(Pmin);
P_min_previous_averaged = sum(Pmin)/current_frame_size;
noise_ps_sum = sum(noise_ps_final_smoothed);
                      
%Track noise ps minimum:
Pmin_current_updated = min(Pmin,noise_ps_smoothed_current);
Pmin_sw = min(Pmin_sw,noise_ps_smoothed_current);

%Decide on where to update noise ps
Decision1 = zeros(current_frame_size,1);
noise_ps_above_certain_threshold_indices1 = find(noise_ps_smoothed_current > gamma1*Pmin_current_updated);
if ~isempty(noise_ps_above_certain_threshold_indices1)
    Decision1(noise_ps_above_certain_threshold_indices1) = 1; 
end
Decision2 = zeros(current_frame_size,1);
noise_ps_above_certain_threshold_indices2 = find(noise_ps_smoothed_current > (Pmin_current_updated + gamma2*P_min_previous_averaged));
if ~isempty(noise_ps_above_certain_threshold_indices2) 
    Decision2(noise_ps_above_certain_threshold_indices2) = 1;
end
logical_mask_where_to_update_noise_ps = Decision1.*Decision2;


%Calculate current total power to P_min previous total power ratio for later Bias
%correction of P_min current to achieve current noise ps:
R_total_power_to_min_total_power_current = noise_ps_sum/(P_min_previous_sum+eps); % Bias factor

%Calculate smoothed total power to P_min previous ratio:
if sum(logical_mask_where_to_update_noise_ps)>0
    R_total_power_to_min_total_power_smoothed = Rmin_old;
else
    R_total_power_to_min_total_power_smoothed = R_total_power_to_min_total_power_smoothing_factor*Rmin_old ...
        + (1-R_total_power_to_min_total_power_smoothing_factor)*R_total_power_to_min_total_power_current;
end

%Calculate final noise ps:
indices_deemed_as_noise = find(logical_mask_where_to_update_noise_ps==0);
indices_deemed_as_not_necessary_noise = find(logical_mask_where_to_update_noise_ps==1);
noise_ps_final_smoothed = zeros(size(current_frame_ps));
%(1). where smoothed noise ps is low it is assumed noise and simply assigned to final noise ps:
noise_ps_final_smoothed(indices_deemed_as_noise) = current_frame_ps(indices_deemed_as_noise);
%(2). where smoothed noise ps is higher then threshold set above we use a
%smoothed version of the bias corrected P_min derived noise ps:
noise_ps_final_smoothed(indices_deemed_as_not_necessary_noise) = R_total_power_to_min_total_power_smoothed * Pmin_current_updated(indices_deemed_as_not_necessary_noise);

%Temporal minimum tracking
%use window to find the minimum
j = j+1;
if j==number_of_frames_to_buildup_spectrum_minimum
    stored_min(:,u1) = Pmin_sw;
    u1 = u1+1;
    if u1==number_of_spectral_minimum_buildup_blocks_to_remember+1; 
        u1=1;
    end
    Pmin_current_updated = min(stored_min,[],2);
    Pmin_sw = noise_ps_smoothed_current;
    j = 0;
end

%Update final parameters:
parameters.alpha_c = alpha_c_noise_ps_smoother_correction_factor_smoothed;
parameters.noise_ps = noise_ps_final_smoothed;
parameters.Rmin_old = R_total_power_to_min_total_power_smoothed;
parameters.Pmin = Pmin_current_updated;
parameters.Pmin_sw = Pmin_sw;
parameters.SmthdP = noise_ps_smoothed_current;
parameters.u1 = u1;
parameters.j = j;
parameters.alpha = alpha_opt_noise_ps_smoothing_factor;
parameters.stored_min = stored_min;
parameters.Decision = logical_mask_where_to_update_noise_ps;


% ----------------------------------------------
function [y] = smoothing(x,win,N)


len=length(x);
win1=win(1:N+1);
win2=win(N+2:2*N+1);
y1=filter(fliplr(win1),[1],x);

x2=zeros(len,1);
x2(1:len-N)=x(N+1:len);

y2=filter(fliplr(win2),[1],x2);

y=(y1+y2); 
