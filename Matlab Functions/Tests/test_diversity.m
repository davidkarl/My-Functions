%parameters:
Fs = 44100;
number_of_seconds = 0.5;
number_of_samples = Fs * number_of_seconds;
f_max_doppler = 45;

%build amplitude vecs:
amplitude_factor1 = 300;
amplitude_factor2 = 30;
amplitude_threshold = 0;
amplitude_vec1 = rayleigh_fading(Fs,number_of_samples,f_max_doppler);
amplitude_vec2 = rayleigh_fading(Fs,number_of_samples,f_max_doppler);
amplitude_vec1(amplitude_vec1<amplitude_threshold) = 0;
amplitude_vec2(amplitude_vec2<amplitude_threshold) = 0;
amplitude_vec1 = amplitude_vec1 * amplitude_factor1;
amplitude_vec2 = amplitude_vec2 * amplitude_factor2;

%sines:
Fc = 12000;
t_vec = my_linspace(0,1/Fs,number_of_samples);
phase_difference = pi/180 * 90;
sine_vec1 = sin(2*pi*Fc*t_vec);
sine_vec2 = sin(2*pi*Fc*t_vec + phase_difference);

%fading sines:
fading_sine1 = amplitude_vec1 .* sine_vec1;
fading_sine2 = amplitude_vec2 .* sine_vec2; 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Add noise:
noise_factor = 0;
fading_sine1 = fading_sine1 + randn(size(amplitude_vec1)).*sqrt(rms(amplitude_vec1)*5) + randn(size(fading_sine1))*noise_factor;
fading_sine2 = fading_sine2 + randn(size(amplitude_vec1)).*sqrt(rms(amplitude_vec2)*5) + randn(size(fading_sine2))*noise_factor;
fading_sine3 = fading_sine1 + fading_sine2 + randn(size(fading_sine1))*noise_factor;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Analytic signals:
analytic_signal_object1 = dsp.AnalyticSignal;
analytic_signal_object2 = dsp.AnalyticSignal;
analytic_signal_object3 = dsp.AnalyticSignal;

%Filters:
%%%SIGNAL FILTER:
%Signal filter parameters:
signal_filter_parameter = 7;
signal_filter_length = 128*8; 
signal_filter_window_type = 'kaiser';
signal_filter_type = 'bandpass';
signal_filter_start_frequency = 250;
signal_filter_stop_frequency = 5000;
signal_filter_window_type_string_array = {'kaiser','hann','hanning','hamming'};
%create filter:
[signal_filter] = get_filter_1D(signal_filter_window_type,signal_filter_parameter,signal_filter_length,Fs,signal_filter_start_frequency,signal_filter_stop_frequency,signal_filter_type);
signal_filter_object1 = dsp.FIRFilter('Numerator',signal_filter.Numerator);
signal_filter_object2 = dsp.FIRFilter('Numerator',signal_filter.Numerator);
signal_filter_object3 = dsp.FIRFilter('Numerator',signal_filter.Numerator);
%%%CARRIER FILTER:
BW = signal_filter_stop_frequency*2;
carrier_filter_parameter = 10;
carrier_filter_length = 128*8; 
carrier_filter_window_type = 'hann';
carrier_filter_start_frequency = Fc-BW/2;
carrier_filter_stop_frequency = Fc+BW/2;
carrier_filter_type = 'bandpass';
carrier_filter_window_type_string_array = {'hann','kaiser','hanning','hamming'};
%create filter:
[carrier_filter] = get_filter_1D(carrier_filter_window_type,carrier_filter_parameter,carrier_filter_length,Fs,carrier_filter_start_frequency,carrier_filter_stop_frequency,carrier_filter_type);
carrier_filter_object1 = dsp.FIRFilter('Numerator',carrier_filter.Numerator);
carrier_filter_object2 = dsp.FIRFilter('Numerator',carrier_filter.Numerator);
carrier_filter_object3 = dsp.FIRFilter('Numerator',carrier_filter.Numerator);


%DEMODULATE:
%filter carriers:
current_large_frame_FM1 = step(carrier_filter_object1,fading_sine1);
current_large_frame_FM2 = step(carrier_filter_object2,fading_sine2);
current_large_frame_FM3 = step(carrier_filter_object3,fading_sine3);

%get analytic signal:
analytic_signal1 = step(analytic_signal_object1,current_large_frame_FM1);
analytic_signal2 = step(analytic_signal_object2,current_large_frame_FM2);
analytic_signal3 = step(analytic_signal_object3,current_large_frame_FM3);

%get FM:
amplitude1 = abs(analytic_signal1.*conj(analytic_signal1));
amplitude2 = abs(analytic_signal2.*conj(analytic_signal2));
amplitude3 = abs(analytic_signal3.*conj(analytic_signal3));
amplitude1 = (amplitude1(2:end)+amplitude1(1:end-1))/2;
amplitude2 = (amplitude2(2:end)+amplitude2(1:end-1))/2;
amplitude3 = (amplitude3(2:end)+amplitude3(1:end-1))/2;
phase1 = angle(analytic_signal1(2:end).*conj(analytic_signal1(1:end-1)));
phase2 = angle(analytic_signal2(2:end).*conj(analytic_signal2(1:end-1)));
phase3 = angle(analytic_signal3(2:end).*conj(analytic_signal3(1:end-1)));

%filter phase:
phase1_filtered = step(signal_filter_object1,phase1);
phase2_filtered = step(signal_filter_object2,phase2);
phase12_combined = (phase1.*amplitude1.^2 + phase2.*amplitude2.^2)./(amplitude1.^2+amplitude2.^2);
phase12_combined_filtered = step(signal_filter_object3,phase12_combined);
phase3_filtered = step(signal_filter_object3,phase3);

subplot(5,1,1)
plot(t_vec(5000:end),fading_sine1(5000:end));
subplot(5,1,2)
plot(phase1_filtered(5000:end));
title(strcat('CHANNEL1:   RMS : ',num2str(rms(phase1_filtered(5000:end)))));
ylim([-0.3,0.3]);
subplot(5,1,3)
plot(phase2_filtered(5000:end));
title(strcat('CHANNEL2:   RMS : ',num2str(rms(phase1_filtered(5000:end)))));
ylim([-0.3,0.3]);
subplot(5,1,4)
plot(phase12_combined_filtered(5000:end));
title(strcat('COMBINED SPLIT POLARIZATIONS:   RMS : ',num2str(rms(phase12_combined_filtered(5000:end)))));
ylim([-0.3,0.3]);
subplot(5,1,5)
plot(phase3_filtered(5000:end));
title(strcat('BOTH POLARIZATIONS CHANNELS:   RMS : ',num2str(rms(phase3_filtered(5000:end)))));
ylim([-0.3,0.3]);






