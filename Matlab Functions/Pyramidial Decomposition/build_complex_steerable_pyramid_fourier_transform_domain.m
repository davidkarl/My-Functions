function [pyramid,index_matrix] = build_complex_steerable_pyramid_fourier_transform_domain(mat_in, cropped_filters, filter_indices, varargin)
% [PYR, PIND] = buildSCFpyrGenPar(IM, croppedFilters, FILTIDX, ...)
%
% This is pyramid building function, which will apply FILTERS to the image
% and give back a pyramid. It is expected that filters be a cell array in
% which the first filter is the hi pass residual and the last filter is the
% lowpass residual
%
% Based on buildSCFpyr in matlabPyrTools
%
% Authors: Neal Wadhwa
% License: Please refer to the LICENCE file
% Date: July 2013
%
 
 
%if(not(isa(filtersD,'distributed')))
%    error('Second argument must be a distributed array of spatial filters\n');
%end


number_of_filters = max(size(cropped_filters));

%Parse optional arguments
p = inputParser;
defaultInputIsFreqDomain = false;
addOptional(p, 'inputFreqDomain', defaultInputIsFreqDomain, @islogical);
parse(p,  varargin{:});
flag_is_frequency_domain = p.Results.inputFreqDomain;


%Return pyramid in the usual format of a stack of column vectors
if flag_is_frequency_domain
    mat_in_fft = mat_in;
else
    mat_in_fft = fftshift(fft2(mat_in)); %DFT of image
end



%pyr = (cell(1,nFilts));
%pind = (cell(1,nFilts));


%Build Pyamid in the Transform Domain:
pyramid = [];
index_matrix = zeros(number_of_filters,2);
for k = 1:number_of_filters
    tempDFT = cropped_filters{k}.*mat_in_fft(filter_indices{k,1}, filter_indices{k,2}); % Transform domain
    current_pyramid_spatial_domain = ifft2(ifftshift(tempDFT));
    %pyr{k} = curResult(:);
    %pind{k} = size(curResult);
    index_matrix(k,:) = size(current_pyramid_spatial_domain);
    pyramid = [pyramid; current_pyramid_spatial_domain(:)];
end
end
