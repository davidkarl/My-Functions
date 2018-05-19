function [actual_filter] = get_filter_1D(filter_name,filter_parameter,N,f_sampling,f_low_cutoff,f_high_cutoff,filter_type)

if strcmp(filter_type,'bandstop')
    filter_type = 'stop';
end
%get window type wanted:
if strcmp(filter_name, 'kaiser')==1
    
    
    window = kaiser(N+1, filter_parameter); %filter_parameter == beta
    
    % Calculate the coefficients using the FIR1 function.
    if strcmp(filter_type,'bandpass') || strcmp(filter_type,'stop')
        coefficients = fir1(N, [f_low_cutoff f_high_cutoff]/(f_sampling/2), filter_type, window, 'scale');
    elseif strcmp(filter_type,'low') || strcmp(filter_type,'lowpass')
        filter_type = 'low';
        coefficients = fir1(N, f_low_cutoff/(f_sampling/2),filter_type,window,'scale');
    elseif strcmp(filter_type,'high') || strcmp(filter_type,'highpass')
        filter_type = 'high';
        coefficients = fir1(N, f_high_cutoff/(f_sampling/2),filter_type,window,'scale');
    end
    actual_filter = dfilt.dffir(coefficients); 
    actual_filter.PersistentMemory = true;
    actual_filter.States = 0;
     
elseif strcmp(filter_name, 'hann')==1
    
    window = hann(N+1,'periodic'); %filter_parameter == beta
    
    % Calculate the coefficients using the FIR1 function.
    if strcmp(filter_type,'bandpass') || strcmp(filter_type,'stop')
        coefficients = fir1(N, [f_low_cutoff f_high_cutoff]/(f_sampling/2), filter_type, window, 'scale');
    elseif strcmp(filter_type,'low') || strcmp(filter_type,'lowpass')
        filter_type = 'low';
        coefficients = fir1(N, f_low_cutoff/(f_sampling/2),filter_type,window,'scale');
    elseif strcmp(filter_type,'high') || strcmp(filter_type,'highpass')
        filter_type = 'high';
        coefficients = fir1(N, f_high_cutoff/(f_sampling/2),filter_type,window,'scale');
    end
    actual_filter = dfilt.dffir(coefficients);
    actual_filter.PersistentMemory = true;
    actual_filter.States = 0;
    
elseif strcmp(filter_name, 'hamming')==1
    
    window = hamming(N+1,'periodic'); %filter_parameter == beta
    
    % Calculate the coefficients using the FIR1 function.
    if strcmp(filter_type,'bandpass') || strcmp(filter_type,'stop')
        coefficients = fir1(N, [f_low_cutoff f_high_cutoff]/(f_sampling/2), filter_type, window, 'scale');
    elseif strcmp(filter_type,'low') || strcmp(filter_type,'lowpass')
        filter_type = 'low';
        coefficients = fir1(N, f_low_cutoff/(f_sampling/2),filter_type,window,'scale');
    elseif strcmp(filter_type,'high') || strcmp(filter_type,'highpass')
        filter_type = 'high';
        coefficients = fir1(N, f_high_cutoff/(f_sampling/2),filter_type,window,'scale');
    end
    actual_filter = dfilt.dffir(coefficients);
    actual_filter.PersistentMemory = true;
    actual_filter.States = 0;
    
elseif strcmp(filter_name, 'hanning')==1
    
    window = hanning(N+1,'periodic'); %filter_parameter == beta
    
    % Calculate the coefficients using the FIR1 function.
    if strcmp(filter_type,'bandpass') || strcmp(filter_type,'stop')
        coefficients = fir1(N, [f_low_cutoff f_high_cutoff]/(f_sampling/2), filter_type, window, 'scale');
    elseif strcmp(filter_type,'low') || strcmp(filter_type,'lowpass')
        filter_type = 'low';
        coefficients = fir1(N, f_low_cutoff/(f_sampling/2),filter_type,window,'scale');
    elseif strcmp(filter_type,'high') || strcmp(filter_type,'highpass')
        filter_type = 'high';
        coefficients = fir1(N, f_high_cutoff/(f_sampling/2),filter_type,window,'scale');
    end
    actual_filter = dfilt.dffir(coefficients);
    actual_filter.PersistentMemory = true;
    actual_filter.States = 0;
    
    
elseif strcmp(filter_name, 'cheb')==1
    
    window = chebwin(N+1,filter_parameter); %filter_parameter == beta
    
    % Calculate the coefficients using the FIR1 function.
    if strcmp(filter_type,'bandpass') || strcmp(filter_type,'stop')
        coefficients = fir1(N, [f_low_cutoff f_high_cutoff]/(f_sampling/2), filter_type, window, 'scale');
    elseif strcmp(filter_type,'low') || strcmp(filter_type,'lowpass')
        filter_type = 'low';
        coefficients = fir1(N, f_low_cutoff/(f_sampling/2),filter_type,window,'scale');
    elseif strcmp(filter_type,'high') || strcmp(filter_type,'highpass')
        filter_type = 'high';
        coefficients = fir1(N, f_high_cutoff/(f_sampling/2),filter_type,window,'scale');
    end
    actual_filter = dfilt.dffir(coefficients);
    actual_filter.PersistentMemory = true;
    actual_filter.States = 0;
    
elseif strcmp(filter_name, 'blackmanharris')==1
    
    window = blackmanharris(N+1); %filter_parameter == beta
    
    % Calculate the coefficients using the FIR1 function.
    if strcmp(filter_type,'bandpass') || strcmp(filter_type,'stop')
        coefficients = fir1(N, [f_low_cutoff f_high_cutoff]/(f_sampling/2), filter_type, window, 'scale');
    elseif strcmp(filter_type,'low') || strcmp(filter_type,'lowpass')
        filter_type = 'low';
        coefficients = fir1(N, f_low_cutoff/(f_sampling/2),filter_type,window,'scale');
    elseif strcmp(filter_type,'high') || strcmp(filter_type,'highpass')
        filter_type = 'high';
        coefficients = fir1(N, f_high_cutoff/(f_sampling/2),filter_type,window,'scale');
    end
    actual_filter = dfilt.dffir(coefficients);
    actual_filter.PersistentMemory = true;
    actual_filter.States = 0;
    
end










