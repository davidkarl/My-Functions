function filtered = apply_ideal_bandpass_to_mat(...
                mat_in, dimension_to_do_filtering, low_cutoff_frequency, high_cutoff_frequency, sampling_rate)
% FILTERED = ideal_bandpassing(INPUT,DIM,WL,WH,SAMPLINGRATE)
% 
% Apply ideal band pass filter on INPUT along dimension DIM.
% 
% WL: lower cutoff frequency of ideal band pass filter
% WH: higher cutoff frequency of ideal band pass filter
% SAMPLINGRATE: sampling rate of INPUT
% 
% Copyright (c) 2011-2012 Massachusetts Institute of Technology, 
% Quanta Research Cambridge, Inc.
%
% Authors: Hao-yu Wu, Michael Rubinstein, Eugene Shih, 
% License: Please refer to the LICENCE file
% Date: June 2012
%

    if (dimension_to_do_filtering > size(size(mat_in),2))
        error('Exceed maximum dimension');
    end

    mat_in_shifted = shiftdim(mat_in,dimension_to_do_filtering-1);
    mat_in_shifted_size = size(mat_in_shifted);
    
    n = mat_in_shifted_size(1);
    dn = size(mat_in_shifted_size,2);
    
    
    frequency_vec = 1:n;
    frequency_vec = (frequency_vec-1)/n*sampling_rate;
    band_pass_logical_mask = frequency_vec > low_cutoff_frequency & frequency_vec < high_cutoff_frequency;
    
    mat_in_shifted_size(1) = 1;
    band_pass_logical_mask = band_pass_logical_mask(:);
    band_pass_logical_mask = repmat(band_pass_logical_mask, mat_in_shifted_size);

    
    F = fft(mat_in_shifted,[],1);
    
    F(~band_pass_logical_mask) = 0;
    
    filtered = real(ifft(F,[],1));
    
    filtered = shiftdim(filtered,dn-(dimension_to_do_filtering-1));
    
end
