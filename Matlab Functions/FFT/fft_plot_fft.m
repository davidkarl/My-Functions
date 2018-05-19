function [] = fft_plot_fft(input_signal,Fs,flag_one_sided,flag_fft_or_spectrum,flag_linear_or_log)

[calculated_fft,f_vec] = fft_calculate_simple_fft(input_signal,Fs,flag_fft_or_spectrum,flag_one_sided);
calculated_fft = abs(calculated_fft);
if flag_linear_or_log==2
   calculated_fft = 10*log10(abs(calculated_fft)); 
end
plot(f_vec,(calculated_fft));


