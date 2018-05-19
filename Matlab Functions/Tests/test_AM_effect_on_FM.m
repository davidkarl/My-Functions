%test AM influence on FM:

Fs = 44100;
Fc = 8000;
number_of_seconds = 2;
N = round(Fs*number_of_seconds);
t_vec = my_linspace(0,1/Fs,N);
amplitude = ones(N,1);
amplitude_SNR = 10;
[amplitude] = add_noise_of_certain_SNR( amplitude, amplitude_SNR , 2, -0.3);
carrier_signal = amplitude .* sin(2*pi*Fc*t_vec);



analytic_signal = hilbert(carrier_signal);
phase_difference = angle(analytic_signal(2:end).*conj(analytic_signal(1:end-1)));
analytic_signal_amplitude = abs(analytic_signal);

subplot(4,1,1)
plot(t_vec,amplitude);
ylim([0,max(amplitude)+0.1]);
xlim([0,number_of_seconds]);
subplot(4,1,2)
plot(t_vec,analytic_signal_amplitude);
ylim([0,max(analytic_signal_amplitude)+0.1]);
xlim([0,number_of_seconds]);
subplot(4,1,3)
plot(t_vec,analytic_signal_amplitude-amplitude)
xlim([0,number_of_seconds]);
subplot(4,1,4)
plot(t_vec(2:end),phase_difference-mean(phase_difference));
xlim([0,number_of_seconds]);




