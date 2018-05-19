function [parameters] = noise_estimation_doblinger(current_frame_ps,parameters)

%         parameters = struct('n',2,'len',len_val,'alpha',0.7,'beta',0.96,'gamma',0.998,'noise_ps',ns_ps,'pxk_old',ns_ps,...
%             'pxk',ns_ps,'pnk_old',ns_ps,'pnk',ns_ps);

frame_counter = parameters.n;
current_frame_size = parameters.len;
raw_ps_smoothing_factor = parameters.alpha;
noise_ps_look_ahead_update_factor = parameters.beta;
noise_ps_smoothing_factor = parameters.gamma;

%Previous power spectrums:
raw_ps_previous = parameters.pxk_old;
noise_ps_previous = parameters.pnk_old;


%Calculate smoothed raw power spectrum:
raw_ps_current = raw_ps_smoothing_factor*raw_ps_previous + (1-raw_ps_smoothing_factor)*current_frame_ps;

%Calculate smoothed noise power spectrum:
noise_ps_current = zeros(size(noise_ps_previous));
indices_where_speech_is_deemed_maybe_present = find(noise_ps_previous <= raw_ps_current);
indices_where_speech_is_deemed_absent = find(noise_ps_previous > raw_ps_current);
noise_ps_current(indices_where_speech_is_deemed_absent) = raw_ps_current(indices_where_speech_is_deemed_absent);
noise_ps_current(indices_where_speech_is_deemed_maybe_present) = noise_ps_smoothing_factor.*noise_ps_previous(indices_where_speech_is_deemed_maybe_present) ...
    + (1-noise_ps_smoothing_factor).*(raw_ps_current(indices_where_speech_is_deemed_maybe_present)-noise_ps_look_ahead_update_factor.*raw_ps_previous(indices_where_speech_is_deemed_maybe_present))./(1-noise_ps_look_ahead_update_factor);


%Update final power spectrums:
raw_ps_previous = raw_ps_current;
noise_ps_previous = noise_ps_current;
noise_ps = noise_ps_current;

%Update final parameters:
parameters.n = frame_counter+1;
parameters.noise_ps = noise_ps;
parameters.pnk = noise_ps_current;
parameters.pnk_old = noise_ps_previous;
parameters.pxk = raw_ps_current;
parameters.pxk_old = raw_ps_previous;

