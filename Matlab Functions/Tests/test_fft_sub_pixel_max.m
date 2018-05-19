%test max frequency find:
%SEEMS DIFFERENCE ISN'T TOO GREATE.
%seems fourier upsampling does most of the work and sub pixel interpolation
%doesn't help much.
%maybe high accuracy kernel based methods can help or maybe neural networks


Fc = 12070;
Fs = 44100;
N = 512;
t_vec = my_linspace(0,1/Fs,N);
A_phase = 0.3; 
A_noise = 0;
y = sin(2*pi*Fc*t_vec + A_phase*randn(N,1)) + A_noise*randn(N,1);


flag_linear_or_log = 2;
flag_fft_or_spectrum = 2;
delta_f = Fs / N;
upsample_factor = 4;
FFT_length = N*upsample_factor;

%calculate:
y_fft = abs(fftshift(fft(y,FFT_length)));
if flag_fft_or_spectrum == 2
   y_fft = y_fft.^2; 
end
if flag_linear_or_log == 2 
   y_fft = 10*log10(y_fft); 
end 
y_fft(1:end/2) = 0;
[peak_value, peak_index] = max(y_fft);
peak_frequency = (peak_index-FFT_length/2) * Fs/FFT_length;
% figure
% plot(y_fft);   

peak_minus = y_fft(peak_index - 1);
peak_plus = y_fft(peak_index + 1);
peak_interpolated = (peak_plus-peak_minus) / (4*peak_value - 2*(peak_plus+peak_minus));
peak_frequency_interpolated = peak_frequency + peak_interpolated*delta_f;
 
e1 = Fc - peak_frequency;
e2 = Fc - peak_frequency_interpolated;
1;
% plot(2*pi*e2*t_vec); 




 