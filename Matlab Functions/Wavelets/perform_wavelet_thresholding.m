function [output_signal_after_thresholding] = perform_wavelet_thresholding( input_signal_before_thresholding, noise_statistics, ...
	threshold_type, threshold_function_type, wavelet_filters, number_of_decomposition_levels)

%note on the parameters
%   input_signal_before_thresholding: data before thresholding
%   noise_statistics: the power spectrum of the noise (i.e., noise statistics), 
%       DFT of the first row of Sigma_N, refer to Eq. (8) in Walden's paper
%   threshold_type: threshold type, scale-dependent Universal ('d'), 
%		scale-independent Universal ('i'), scale-dependent SURE ('ds'), 
%		scale-independent SURE ('is'), or scale-dependent Generalized 
%		Corss-Validation ('dg')
%   threshold_function_type: threshold function type: soft ('s') or hard ('h');
%   wavelet_filters: wavelet low pass and high pass decomposition/reconstruction filters [lo_d, hi_d, lo_r, hi_r]
%       the 1st row is lo_d, the 2nd row is hi_d, the 3rd row is lo_r, and the 4th row is hi_r
%   decomposition_level: is the decomposition level

%   output_signal_after_thresholding: data after thresholding

data_size = size( input_signal_before_thresholding);  
input_signal_before_thresholding = make_row(input_signal_before_thresholding);
noise_statistics = make_row(noise_statistics);

input_signal_number_of_samples = length(input_signal_before_thresholding); 
wavelet_filter_FFT_size = 2^nextpow2(input_signal_number_of_samples);

%Get the low pass and high pass decomposition/reconstruction filters from wavelet_filters:
low_pass_decomposition_filter = wavelet_filters(1,:);   %low pass decomposition filter/ scaling filter
high_pass_decomposition_filter = wavelet_filters(2,:);   %high pass decomposition filter/ wavelet filter
low_pass_reconstruction_filter = wavelet_filters(3,:);   %low pass reconstruction filter/ scaling filter
high_pass_reconstruction_filter = wavelet_filters(4,:);   %high pass reconstruction filter/ wavelet filter

%Refer to pp. 3155 in Walden's paper
H = zeros( number_of_decomposition_levels, wavelet_filter_FFT_size);
H(1,:) = fft(high_pass_decomposition_filter, wavelet_filter_FFT_size);   %frequency response of wavelet filter
G(1,:) = fft(low_pass_decomposition_filter, wavelet_filter_FFT_size);  %frequency response of scaling filter
for i = 2 : number_of_decomposition_levels-1
    G(i,:) = G(1,1+rem(2^(i-1)*[0:wavelet_filter_FFT_size-1] , wavelet_filter_FFT_size));
end
for j = 2: number_of_decomposition_levels
    H(j,:) = prod( [G(1:j-1,:) ; H(1,1+rem(2^(j-1)*[0: wavelet_filter_FFT_size-1] , wavelet_filter_FFT_size))], 1);
end

%Decompose input signal into number_of_decomposition_levels levels:
[y_coeff, len_info] = wavedec( input_signal_before_thresholding, number_of_decomposition_levels, low_pass_decomposition_filter, high_pass_decomposition_filter);
% --where y_coeff contains the coefficients and len_info contains the length information
% --different segments of y_coeff correspond approximation and detail coefficients;
% --length of len_info should be q_0+ 2


%Process according to "threshold_type":
%-------with 'd'--scale-dependent thresholding, threshold has to be computed for each level
%-------with 'i'--scale-independent thresholding, threshold is set to a fixed level

if threshold_type == 'i' %scale-independent universal thresholding
    sigma_square = mean(noise_statistics);
    wavelet_threshold = sqrt( 2*sigma_square*log(wavelet_filter_FFT_size) ) ;  
    y_coeff(len_info(1)+1:end) = wthresh( y_coeff(len_info(1)+1:end), threshold_function_type , wavelet_threshold);
    
elseif threshold_type == 'd' %scale-dependent universal thresholding
    %first we need to compute the energy level of each scale from j= 1:q_0
    for i = 1:number_of_decomposition_levels   %refer to Eq. (9) in Walden's paper
        sigma_j_square(i) = mean( noise_statistics .* (abs(H(i,:)).^ 2) , 2);   %average along the row          
    end

    for i = 2:number_of_decomposition_levels+1    %thresholding for each scale
        start_point = sum( len_info(1:i-1), 2 ) + 1; %starting point
        end_point = start_point + len_info(i) - 1;
        wavelet_threshold = sqrt( 2*sigma_j_square(number_of_decomposition_levels-i+2)*log(len_info(i)) );
        y_coeff(start_point:end_point) = wthresh( y_coeff(start_point:end_point), threshold_function_type, wavelet_threshold);
    end
    
elseif threshold_type == 'ds' %scale-dependent SURE thresholding
    %first estimate the standard deviation of the different levels (i think):
    sigma_j = wnoisest( y_coeff, len_info, 1: number_of_decomposition_levels);  
    
    for i = 2: number_of_decomposition_levels+1    %thresholding for each scale
        start_point = sum(len_info(1:i-1),2) + 1; %starting point
        end_point = start_point + len_info(i) - 1;    %ending point
        if sigma_j(number_of_decomposition_levels-i+2) < sqrt(eps)*max(y_coeff(start_point:end_point));
            wavelet_threshold = 0;
        else
            wavelet_threshold = sigma_j(number_of_decomposition_levels-i+2) * thselect(y_coeff(start_point:end_point) ...
                              / sigma_j(number_of_decomposition_levels-i+2), 'heursure');
        end
        y_coeff(start_point:end_point) = wthresh( y_coeff(start_point:end_point), threshold_function_type, wavelet_threshold);
    end
    
elseif threshold_type == 'dn' %new risk function defined in Xiao-ping Zhang's paper
    
    sigma_j= wnoisest( y_coeff, len_info, 1:number_of_decomposition_levels);  
    sigma_j_square = sigma_j.^ 2;
    
    for i= 2: number_of_decomposition_levels+ 1    %thresholding for each scale
        start_point = sum(len_info(1:i-1),2) + 1; %starting point
        end_point = start_point + len_info(i)- 1;    %ending point  
        
        if sigma_j(number_of_decomposition_levels-i+2) < sqrt(eps)* max(y_coeff(start_point:end_point));
            wavelet_threshold= 0;
        else    
            
            %based on some evidece, the following theme let thre vary with SNR
            %   with ultra low SNR indicating low probability of signal presence, 
            %       hence using universal threshold
            %   and very high SNR indicates high probability of signal presence,
            %       hence using SURE threshold
            
            thre_max = sigma_j(number_of_decomposition_levels-i+2) * sqrt(2*log(len_info(i))); %thre with SNRlog< -5dB
            thre_min = sigma_j(number_of_decomposition_levels-i+2) * fminbnd(@riskfunc, 0, sqrt(2*log(end_point-start_point+1)), ...
                optimset( 'MaxFunEvals',1000,'MaxIter',1000), ...
                y_coeff( start_point: end_point)/ sigma_j( number_of_decomposition_levels- i+ 2), 3);   %thre with SNRlog> 20dB
            slope = (thre_max-thre_min)/25;
            thre_0 = thre_min + 20*slope;        

            SNRlog = 10*log10( mean( max( y_coeff(start_point:end_point).^2 / sigma_j_square(number_of_decomposition_levels-i+2) - 1, 0) ) );            
            if SNRlog >= 20
                wavelet_threshold = thre_min;  %actually this corresponds to SURE threshold
            elseif (SNRlog<20) && (SNRlog>=-5)
                wavelet_threshold = thre_0 - SNRlog*slope;
            else
                wavelet_threshold = thre_max;   %this corresponds to oversmooth threshold
            end
        end 
        
        y_coeff(start_point:end_point) = wthresh( y_coeff(start_point:end_point), threshold_function_type, wavelet_threshold);
        
    end

elseif threshold_type == 'dg' %scale-dependent Generalized Cross Validation thresholding
    
    for i= 2: number_of_decomposition_levels+ 1    %thresholding for each scale
        start_point= sum( len_info( 1: i- 1), 2)+ 1; %starting point
        end_point= start_point+ len_info( i)- 1;    %ending point       
        [y_coeff(start_point:end_point), wavelet_threshold] = mingcv( y_coeff(start_point:end_point), threshold_function_type);
    end   
   
else 
    error( 'wrong thresholding type');
end

%Reconstruct the thresholded coefficients
output_signal_after_thresholding = waverec( y_coeff, len_info, low_pass_reconstruction_filter, high_pass_reconstruction_filter);

if data_size(1)>1 
    output_signal_after_thresholding = output_signal_after_thresholding'; 
end
