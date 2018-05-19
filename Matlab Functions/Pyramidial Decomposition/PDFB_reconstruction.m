function x = PDFB_reconstruction(subband_images_cell_vec, laplacian_pyramid_filter_name, directional_filter_bank_name)
% PDFBREC   Pyramid Directional Filterbank Reconstruction
%
%	x = pdfbrec(y, pfilt, dfilt)
%
% Input:
%   y:	    a cell vector of length n+1, one for each layer of 
%       	subband images from DFB, y{1} is the low band image
%   pfilt:  filter name for the pyramid
%   dfilt:  filter name for the directional filter bank
%
% Output:
%   x:      reconstructed image
%
% See also: PFILTERS, DFILTERS, PDFBDEC

n = length(subband_images_cell_vec) - 1;
if n <= 0
    x = subband_images_cell_vec{1};
    
else
    % Recursive call to reconstruct the low band
    xlo = PDFB_reconstruction(subband_images_cell_vec(1:end-1), laplacian_pyramid_filter_name, directional_filter_bank_name);
    
    % Get the pyramidal filters from the filter name
    [lowpass_analysis_filter, lowpass_synthesis_filter] = get_filters_for_laplacian_pyramid(laplacian_pyramid_filter_name);
    
    % Process the detail subbands
    if length(subband_images_cell_vec{end}) ~= 3
        % Reconstruct the bandpass image from DFB
        
        % Decide the method based on the filter name
        switch directional_filter_bank_name        
            case {'pkva6', 'pkva8', 'pkva12', 'pkva'}	
                % Use the ladder structure (much more efficient)
                xhi = DFB_reconstruction_ladder_structure(subband_images_cell_vec{end}, directional_filter_bank_name);
                
            otherwise	
                % General case
                xhi = DFB_reconstruction(subband_images_cell_vec{end}, directional_filter_bank_name); 
        end
        
        x = laplacian_pyramid_reconstruction(xlo, xhi, lowpass_analysis_filter, lowpass_synthesis_filter);
   
    else    
        % Special case: length(y{end}) == 3
        % Perform one-level 2-D critically sampled wavelet filter bank
        x = wavelet_filterbank_2D_reconstruction(xlo, subband_images_cell_vec{end}{1}, subband_images_cell_vec{end}{2}, subband_images_cell_vec{end}{3}, lowpass_analysis_filter, lowpass_synthesis_filter);
    end
end