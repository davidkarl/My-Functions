%test speech probability median filtering

%get sound:
a=wavread('C:\Users\master\Desktop\matlab\SHIR ENHANCEMENT\1500m, reference loudness=0dBspl, speakers2, laser power=7.8 , Fs=4935Hz, channel1.wav');
%buffer and get fft, magnitude and angle:
samples_per_frame = 512;
overlap_samples_per_frame = 256;
non_overlapping_samples_per_frame = samples_per_frame-overlap_samples_per_frame;
b=buffer(a,samples_per_frame,overlap_samples_per_frame);
cola_window = get_approximately_COLA_window(samples_per_frame,overlap_samples_per_frame);
b_fft=fft(b);
b_mag=abs(fft(b));
b_angle=angle(b_fft);
%get log spectrum:
c_log=10*log10(b_mag.^2);
%MEDIAN FILTER:
c_log_filt = medfilt2(c_log,[3,3]);
%presenct result:
% figure(1);
% subplot(2,1,1);
% imagesc(c_log);
% subplot(2,1,2);
% imagesc(c_log_filt)
%use exponent to get linear and filtered fft magnitude:
c_log_filt_exp = exp(c_log_filt/20);
% figure(2)
% subplot(2,1,1);
% imagesc(b_mag);
% subplot(2,1,2);
% imagesc(c_log_filt_exp);
%use original phase and median filtered fft magnitude for new sound:

new_values_mat = change_values_range(min(c_log_filt_exp,3),10^(-30/10),1);
filtered=c_log_filt_exp.*exp(1i*b_angle);
filtered=b_fft.*new_values_mat.*exp(1i*b_angle);
filtered_ifft=real(ifft(filtered));
[ vec ] = frames2vec( filtered_ifft, non_overlapping_samples_per_frame,'cols',cola_window);
%sound:
soundsc(vec,4935)





