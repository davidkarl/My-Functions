function parameters = noise_estimation_hirsch(current_frame_ps,parameters)

%         parameters = struct('n',2,'len',len_val,'as',0.85,'as1',0.7,'beta',1.5,'omin',1.5,'noise_ps',ns_ps,'P',ns_ps);

%Get initial parameters:
raw_ps_smoothing_factor = parameters.as;
speech_presence_threshold = parameters.beta;

noise_ps = parameters.noise_ps;
raw_ps_smoothed = parameters.P;

%Calculate smoothed raw ps:
raw_ps_smoothed = raw_ps_smoothing_factor*raw_ps_smoothed + (1-raw_ps_smoothing_factor)*current_frame_ps;

%Update final noise ps only where speech is deemed absent:
indices_where_speech_is_deemed_absent = find( raw_ps_smoothed./noise_ps < speech_presence_threshold);
noise_ps(indices_where_speech_is_deemed_absent) = raw_ps_smoothing_factor*noise_ps(indices_where_speech_is_deemed_absent) ...
    +(1-raw_ps_smoothing_factor)*raw_ps_smoothed(indices_where_speech_is_deemed_absent);

%Update final parameters:
parameters.P = raw_ps_smoothed;
parameters.noise_ps = noise_ps;