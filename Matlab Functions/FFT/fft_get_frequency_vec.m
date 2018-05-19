function [f_vec] = fft_get_frequency_vec(N,Fs,flag_one_sided)
% Fs=1/dt;
f_vec = (Fs/N)*[-floor((N-1)/2) :1: ceil((N-1)/2)];
if flag_one_sided==1
   f_vec=f_vec(floor(length(f_vec)/2)+mod(N,2):end);
end


