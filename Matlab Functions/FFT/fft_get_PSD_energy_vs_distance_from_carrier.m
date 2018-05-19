function [energy_vec,frequency_offset_vec] = fft_get_PSD_energy_vs_distance_from_carrier(signal,Fs,Fc,BW_around_carrier,flag_got_signal_fft_or_PSD,flag_get_fft_or_PSD,flag_get_Fc_or_search)

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

%get frequency vec:
[f_vec] = fft_get_frequency_vec(FFT_length,Fs,1);
delta_f = f_vec(2)-f_vec(1);
if flag_get_Fc_or_search==1
    Fc_index = fft_frequency_to_index(FFT_length,Fs,Fc,1);
else
    [Fc_max_value,Fc_index] = max(signal);
end


%search for max:
flag=1;
counter=0;
while flag==1
    if Fc_index+counter>length(signal) || Fc_index-counter<1
        flag=0;
    elseif f_vec(Fc_index+counter)>Fc+BW_around_carrier || f_vec(Fc_index-counter)<Fc-BW_around_carrier
        flag=0;
    else
        energy_vec(counter+1) = sum(signal(Fc_index-counter:Fc_index+counter));
        frequency_offset_vec(counter+1) = counter*delta_f;
        counter=counter+1;
    end
end















