function res = reconstruct_complex_steerable_pyramid_fourier_transform_domain(...
                                pyramid, index_matrix, sub_sampled_filters, filter_indices, varargin)
% RES = reconSCFpyrGen(PYR, PIND, FILTERS, TWIDTH)
%
% Reconstruct image from its steerable pyramid representation, in the Fourier
% domain, as created by buildSCFpyrGen.
%
% Based on buildSCFpyr in matlabPyrTools
%
% Authors: Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: July 2013
%

% Parse optional arguments
    p = inputParser;
    number_of_filters = max(size(sub_sampled_filters));

    defaultComplex = true; %Filters only span half plane
    defaultOutputIsFreqDomain = false;

    addOptional(p, 'complex', defaultComplex, @islogical);
    addOptional(p, 'outputFreqDomain', defaultOutputIsFreqDomain, @islogical);

    parse(p,  varargin{:});
    flag_is_complex = p.Results.complex;
    flag_is_frequency_domain = p.Results.outputFreqDomain;

    %Return pyramid in the usual format of a stack of column vectors:
    current_pyramid_fft = zeros(index_matrix(1,:));
    N = 1;
    for k = 1:number_of_filters
        band_values = get_pyramid_subband(pyramid,index_matrix,k);    
        if and( and(flag_is_complex,k ~= 1) , k~=number_of_filters)       
            tempDFT = 2*fftshift(fft2(band_values));
        else
            tempDFT = fftshift(fft2(band_values));
        end    
        tempDFT = tempDFT .* sub_sampled_filters{k};

        current_pyramid_fft(filter_indices{k,1}, filter_indices{k,2}) = ...
                                current_pyramid_fft(filter_indices{k,1}, filter_indices{k,2}) + tempDFT;    
    end
    if flag_is_frequency_domain
        res = current_pyramid_fft;
    else
        res = real(ifft2(ifftshift(current_pyramid_fft)));
    end
end
