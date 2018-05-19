function [selected_elements,selected_frequencies] = fft_get_certain_elements(signal,Fs,flag_got_signal_fft_or_PSD,flag_get_fft_or_PSD,frequency1,frequency2)

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
        signal = signal;
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

%find center element and get two sided frequency vec:
center_index = floor(length(signal)/2)+mod(length(signal),2);
f_vec = fft_get_frequency_vec(length(signal),Fs,0);

%get fft between specified frequency:
start_frequency = frequency1;
stop_frequency = frequency2;
start_index = center_index + ceil(start_frequency/(Fs/2) * (FFT_length/2)+eps); %==round(start_frequency/Fs*FFT_length)
stop_index = center_index + floor(stop_frequency/(Fs/2) * (FFT_length/2)-eps);

if start_index<=0
    start_index=1;
end
if stop_index>=length(signal)
    stop_index=length(signal);
end

selected_elements = signal(start_index:stop_index);
selected_frequencies = f_vec(start_index:stop_index);



