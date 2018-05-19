function [peak_value,peak_index,peak_frequency] = fft_search_max_in_fft(signal,Fs,flag_got_signal_fft_or_PSD,flag_get_fft_or_PSD,frequency_bins)
%assign scale parameters:
FFT_length = length(signal); %double-sides fft length
one_point_in_Hz = (Fs)/(FFT_length);


if flag_got_signal_fft_or_PSD==1
    %got raw signal:
    if flag_get_fft_or_PSD==1
        %get fft:
        signal = abs(fftshift(fft(signal)));
    elseif flag_get_fft_or_PSD==2
        %get PSD:
        signal = abs(fftshift(fft(signal))).^2;
    end 
elseif flag_got_signal_fft_or_PSD==2
    %got fft:
    if flag_get_fft_or_PSD==1
        %do nothing because it's already fft:
        signal = abs(fftshift(signal));
    elseif flag_get_fft_or_PSD==2
        %get PSD from fft:
        signal = abs(fftshift(signal)).^2;
    end
end 

%make one-sided:
signal = signal(floor(end/2)+mod(length(signal),2):end);

%search for max:
peak_value = zeros(size(frequency_bins,1),1);
peak_index = zeros(size(frequency_bins,1),1);
peak_frequency = zeros(size(frequency_bins,1),1);
for k=1:size(frequency_bins(1,:))
    start_frequency = frequency_bins(k,1);
    stop_frequency = frequency_bins(k,2);
     
    %one sided indices:
    start_index = max(round(start_frequency/(Fs/2) * (FFT_length/2)),1); %==round(start_frequency/Fs*FFT_length)
    stop_index = min(round(stop_frequency/(Fs/2) * (FFT_length/2)),length(signal));
    
    %find max:
    [peak_value(k),peak_index(k)] = max(signal(start_index:stop_index));
    peak_index(k) = peak_index(k) + start_index-1 - (~mod(FFT_length,2)); %one sided, the last term is a patch for 
                                                                          %an even FFT_length. actually needs to be fixed down
    peak_frequency(k) = peak_index(k)*(Fs/FFT_length);
    
    %make peak index relevant for two-sided FFT:
    peak_index(k) = peak_index(k) + (ceil(FFT_length/2)+(1-mod(FFT_length,2)))-1;
end




