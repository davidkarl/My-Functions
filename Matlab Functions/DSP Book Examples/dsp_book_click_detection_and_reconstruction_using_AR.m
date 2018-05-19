%dsp book click detection using AR estimation

%original signal (order 10-AR)
a=[1,-1.6507,0.6711,-0.1807,0.6130,-0.6085,0.3977,-0.611,0.5412,0.1321,-0.2393];
AR_model_order=length(a);
number_of_samples=500;
white_noise=randn(1,number_of_samples);
signal_using_AR_process=filter(1,a,white_noise);
signal_rms=sqrt(signal_using_AR_process*signal_using_AR_process'/number_of_samples);

%NBCRAC clicks with an amplitude +-1.5 srms
number_of_clicks=5;
click_indices=[73,193,249,293,422];
click_amplitudes=1.5*signal_rms*(2*round(rand(1,number_of_clicks))-1);
signal_with_clicks=signal_using_AR_process;
signal_with_clicks(click_indices)=signal_using_AR_process(click_indices)+click_amplitudes;
subplot(3,1,1); plot(signal_using_AR_process); grid
subplot(3,1,2); plot(signal_with_clicks); grid;

%detection of the clicks using matched filtering:
[aest,process_noise_variance_estimate]=xtoa(signal_with_clicks,AR_model_order); %estimation of the AR
process_noise_or_residual_estiamte=filter(aest,1,signal_with_clicks); %whitening: estimation of the residual
matched_filtering_result_to_emphasize_clicks=filter(aest(AR_model_order:-1:1),1,process_noise_or_residual_estiamte); %matched filtering


subplot(3,1,3); plot(matched_filtering_result_to_emphasize_clicks); grid
bla = rms(matched_filtering_result_to_emphasize_clicks);
estimated_process_rms=sqrt(process_noise_variance_estimate*aest'*aest);
number_of_sigmas_to_define_threshold=3;
click_threshold=number_of_sigmas_to_define_threshold*estimated_process_rms;
indices_above_click_threshold=find(abs(matched_filtering_result_to_emphasize_clicks)>click_threshold); %threshold
indices_above_click_threshold=indices_above_click_threshold-AR_model_order; %filter delay
number_of_indices_above_click_threshold=length(indices_above_click_threshold);

%extraction of the maxima (3 samples from each other)
distance_between_clicks_vec=indices_above_click_threshold-[0,indices_above_click_threshold(1:number_of_indices_above_click_threshold-1)];
indices_within_click_position_indices_vec_3_samples_apart=find(distance_between_clicks_vec>3);
number_of_clicks_3_samples_apart=length(indices_within_click_position_indices_vec_3_samples_apart);
indices_within_click_position_indices_vec_3_samples_apart=[indices_within_click_position_indices_vec_3_samples_apart,number_of_indices_above_click_threshold+1];

for ii=1:number_of_clicks_3_samples_apart
   t1=indices_above_click_threshold(indices_within_click_position_indices_vec_3_samples_apart(ii));
   t2=indices_above_click_threshold(indices_within_click_position_indices_vec_3_samples_apart(ii+1)-1);
   [click_values(ii),im]=max(matched_filtering_result_to_emphasize_clicks(t1:t2));
   estimated_click_positions(ii)=im+t1;
end


%dsp book signal reconstruction from both sides of click
lsig=length(signal_with_clicks);
tps=[0:lsig-1];
signal_with_clicks=signal_with_clicks(:);
ell=estimated_click_positions-floor(AR_model_order/2);
X0=signal_with_clicks(ell-AR_model_order:ell-1);
X1=signal_with_clicks(ell+AR_model_order:ell+AR_model_order+AR_model_order-1);
colT=[aest(AR_model_order);zeros(AR_model_order+K-1,1)];
ligT=[aest(AR_model_order:-1:1)', zeros(1,AR_model_order+AR_model_order)];
T=toplitz(colT,ligT);
A0=T(:,1:K);
B=T(:,K+1:K+AR_model_order);
A1=T(:,K+AR_model_order+1:2*K+AR_model_order);
X=A0*X0+A1*X1;

%solve the system:
Y=-B\X;
sigr=signal_with_clicks;
sigr(ell:ell+AR_model_order-1)=Y;
plot(tps,signal_with_clicks,'-r',tps,s,'b',tps,sigr,':y');
grid;















