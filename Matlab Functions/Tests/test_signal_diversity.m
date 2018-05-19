%test signal diversity:

[audio_vec,Fs] = read_wav_file_full_options('feynman.wav',10,30,44100,0);

t_vec = my_linspace(0,1/Fs,Fs*10);
audio_vec = audio_vec(1:length(t_vec));
Fc = 12000;
Fm = 2000;
m_sine = 0;
m_audio = 0.4;

% clear sound;
Noise = 4;
low_frequency_noise = 50;
high_frequency_noise = 51;
filter1_noise = get_filter_1D('kaiser',10,2000,Fs,low_frequency_noise,high_frequency_noise,'lowpass');
filter2_noise = get_filter_1D('kaiser',10,2000,Fs,low_frequency_noise,high_frequency_noise,'lowpass');
noise1 = Noise*randn(length(t_vec),1);
noise2 = Noise*randn(length(t_vec),1);
noise1 = filter(filter1_noise.Numerator,1,noise1);
noise2 = filter(filter2_noise.Numerator,1,noise2);
A1 = 1 + noise1;
A2 = 1 + noise2;
y1 = A1.*sin(2*pi*Fc*t_vec + m_sine*sin(2*pi*Fm*t_vec) + m_audio*audio_vec);
y2 = A2.*sin(2*pi*Fc*t_vec + 0*randn(1) + m_sine*sin(2*pi*Fm*t_vec) + m_audio*audio_vec);
noise_additive = 0.1;
noise3 = noise_additive*randn(length(y1),1);
noise4 = noise_additive*randn(length(y2),1);
y1 = y1 + noise3;
y2 = y2 + noise4;

a1 = dsp.AnalyticSignal;
a2 = dsp.AnalyticSignal;
low_frequency = 12000-5000;
high_frequency = 12000+5000;
filter1 = get_filter_1D('kaiser',10,2000,Fs,low_frequency,high_frequency,'bandpass');
filter2 = get_filter_1D('kaiser',10,2000,Fs,low_frequency,high_frequency,'bandpass');
y1 = filter(filter1.Numerator,1,y1);
y2 = filter(filter2.Numerator,1,y2);
y1 = y1(2500:end);
y2 = y2(2500:end);

b1 = step(a1,y1);
b2 = step(a2,y2);
b1 = b1.*exp(-1i*2*pi*Fc*t_vec(2500:end));
b2 = b2.*exp(-1i*2*pi*Fc*t_vec(2500:end));
b1 = b1(2:end).*conj(b1(1:end-1));
b2 = b2(2:end).*conj(b2(1:end-1));

c1 = angle(b1);
c2 = angle(b2);
% c1 = cumsum(c1);
% c2 = cumsum(c2);
% c1 = diff(c1);
% c2 = diff(c2);
d1 = abs(b1);
d2 = abs(b2);
% d1 = d1(2:end);
% d2 = d2(2:end);

c1 = c1(1000:end);
c2 = c2(1000:end);
d1 = d1(1000:end);
d2 = d2(1000:end);


moment = 1;
weighted_signal = (d1.^moment.*c1 + d2.^moment.*c2)./(d1.^moment+d2.^moment);
d3 = (d1.^moment + d2.^moment).^(1/moment);
% c1 = cumsum(c1);
% c2 = cumsum(c2);
% weighted_signal = cumsum(weighted_signal);

soundsc(cumsum(c1),Fs);
% soundsc(cumsum(c2),Fs);
% soundsc(cumsum(weighted_signal),Fs);
% clear sound;

start = 5000;
% stop = 6000;
% c1 = c1(start:stop);
% c2 = c2(start:stop);
% weighted_signal = weighted_signal(start:stop);
% 
% figure;
% plot(c1);
% figure;
% plot(c2);
% figure;
% plot(weighted_signal);

weighted_signal_smoothed = smooth_lsq_minimum_iterations_with_preweights2(weighted_signal,d3,4,3);
% soundsc(cumsum(weighted_signal_smoothed),Fs);
% clear sound;
% figure;
% plot(weighted_signal_smoothed);




