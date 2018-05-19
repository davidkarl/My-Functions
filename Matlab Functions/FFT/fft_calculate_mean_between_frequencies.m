function [mean_vec] = fft_calculate_mean_between_frequencies(signal,Fs,flag_got_signal_fft_or_PSD,flag_get_fft_or_PSD,flag_each_or_total_mean,frequency_bins)
%assign scale parameters:
FFT_length = length(signal); %double-sides fft length
one_point_in_Hz = (Fs)/(FFT_length);


if flag_got_signal_fft_or_PSD==1
    %got raw signal:
    if flag_get_fft_or_PSD==1
        %get fft:
        signal = abs((fftshift(fft(signal))));
    elseif flag_get_fft_or_PSD==2
        %get PSD:
        signal = abs(fftshift(fft(signal))).^2;
    end 
elseif flag_got_signal_fft_or_PSD==2
    %got fft:
    if flag_get_fft_or_PSD==1
        %do nothing because it's already fft:
        signal = abs(signal);
    elseif flag_get_fft_or_PSD==2
        %get PSD from fft:
        signal = abs(signal).^2;
    end
elseif flag_got_signal_fft_or_PSD==3
   %got PSD:
   if flag_get_fft_or_PSD==1
      %got PSD get fft:
      signal = sqrt(signal); 
   elseif flag_get_fft_or_PSD==2
      %got PSD get PSD:
      signal=signal;
   end
end


%make one-sided:
signal = signal(floor(end/2)+mod(length(signal),2):end);

%search for max:
mean_vec = zeros(size(frequency_bins,1),1);
number_of_elements = zeros(size(frequency_bins,1),1);
for k=1:size(frequency_bins,1)
    start_frequency = frequency_bins(k,1);
    stop_frequency = frequency_bins(k,2);
    
    start_index = ceil(start_frequency/(Fs/2) * (FFT_length/2)+eps); %==round(start_frequency/Fs*FFT_length)
    stop_index = floor(stop_frequency/(Fs/2) * (FFT_length/2)-eps);
    if start_index<=0
       start_index=1; 
    end
    if stop_index>=length(signal)
       stop_index=length(signal); 
    end
    number_of_elements(k) = length(signal(start_index:stop_index));
    mean_vec(k) = mean(signal(start_index:stop_index));
end

if flag_each_or_total_mean==2
   %get mean over all frequency bins:
   mean_vec = (sum(mean_vec.*number_of_elements)/sum(number_of_elements)); 
end



