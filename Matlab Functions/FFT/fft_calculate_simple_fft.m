function [calculated_fft,f_vec] = fft_calculate_simple_fft(signal,Fs,flag_fft_or_spectrum,flag_one_sided)
%calculate_simple_fft:
N=length(signal);
f_vec = (Fs/N)*[-floor((N-1)/2) :1: ceil((N-1)/2)];
calculated_fft = fftshift(fft(signal));

if flag_one_sided==1
    f_vec = f_vec(floor(N/2)+mod(N,2):end);
    calculated_fft = calculated_fft(floor(N/2)+mod(N,2):end);
end

if flag_fft_or_spectrum==2
   calculated_fft = abs(calculated_fft).^2; 
end

% if flag_plot==1
%     plot(f_vec,calculated_fft);
% end
