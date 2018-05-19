function [actual_filter] = get_filter_1D_multiband(window_name,filter_parameter,N,Fs,frequency_band_edges,passbands_amplitudes)

%normalize frequency vec:
normalized_frequency_vec = frequency_band_edges/(Fs/2);
final_filter = 0;
for k=1:length(passbands_amplitudes)
    
    f_low_cutoff = frequency_band_edges(k,1);
    f_high_cutoff = frequency_band_edges(k,2);
    
    if strcmp(window_name, 'kaiser')==1
        
        window = kaiser(N+1, filter_parameter); %filter_parameter == beta
        
        % Calculate the coefficients using the FIR1 function.
        if strcmp(filter_type,'bandpass') || strcmp(filter_type,'stop')
            coefficients = fir1(N, [f_low_cutoff f_high_cutoff]/(Fs/2), filter_type, window, 'scale');
        end
        actual_filter = dfilt.dffir(coefficients);
        actual_filter.PersistentMemory = true;
        actual_filter.States = 0;
        
    elseif strcmp(window_name, 'hann')==1
        
        window = hann(N+1); %filter_parameter == beta
        
        % Calculate the coefficients using the FIR1 function.
        if strcmp(filter_type,'bandpass') || strcmp(filter_type,'stop')
            coefficients = fir1(N, [f_low_cutoff f_high_cutoff]/(Fs/2), filter_type, window, 'scale');
        end
        actual_filter = dfilt.dffir(coefficients);
        actual_filter.PersistentMemory = true;
        actual_filter.States = 0;
        
    elseif strcmp(window_name, 'hamming')==1
        
        window = hamming(N+1); %filter_parameter == beta
        
        % Calculate the coefficients using the FIR1 function.
        if strcmp(filter_type,'bandpass') || strcmp(filter_type,'stop')
            coefficients = fir1(N, [f_low_cutoff f_high_cutoff]/(Fs/2), filter_type, window, 'scale');
        end
        actual_filter = dfilt.dffir(coefficients);
        actual_filter.PersistentMemory = true;
        actual_filter.States = 0;
        
    elseif strcmp(window_name, 'blackmanharris')==1
        
        window = blackmanharris(N+1); %filter_parameter == beta
        
        % Calculate the coefficients using the FIR1 function.
        if strcmp(filter_type,'bandpass') || strcmp(filter_type,'stop')
            coefficients = fir1(N, [f_low_cutoff f_high_cutoff]/(Fs/2), filter_type, window, 'scale');
        end
        actual_filter = dfilt.dffir(coefficients);
        actual_filter.PersistentMemory = true;
        actual_filter.States = 0;
    end
    final_filter = final_filter + actual_filter*passsbands_amplitudes(k);
    
end












