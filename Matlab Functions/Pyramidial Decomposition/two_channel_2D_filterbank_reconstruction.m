function x = two_channel_2D_filterbank_reconstruction(subband_mat1, ...
                                                      subband_mat2, ...
                                                      reconstruction_filter1, ...
                                                      reconstruction_filter2, ...
                                                      type1, ...
                                                      type2, ...
                                                      extention_mode_string, ...
                                                      flag_interpolation)
% FBREC   Two-channel 2D Filterbank Reconstruction
%
%	x = fbrec(y0, y1, h0, h1, type1, type2, [extmod], [intp])
%
% Input:
%	y0, y1:	two input subband images
%	h0, h1:	two reconstruction 2D filters
%	type1:	'q', 'p' or 'pq' for selecting quincunx or parallelogram
%		upsampling matrix
%	type2:	second parameter for selecting the filterbank type
%		If type1 == 'q' then type2 is one of {'1r', '1c', '2r', '2c'}
%		If type1 == 'p' then type2 is one of {1, 2, 3, 4}
%			Those are specified in QUP and PUP
%		If type1 == 'pq' then same as 'p' except that
%		the paralellogram matrix is replaced by a combination 
%		of a quincunx and a resampling matrices
%	extmod:	[optional] extension mode (default is 'per')
%   intp: [optional] 0, 1: yes/no interpolation (default is no) 
%       this option is used in case of interpolation the subband image
%       using diamond interpolation filter. The task of the interpolation
%       is to estimate the subband coeffs correspond to the shifted input
%       in case of interpolation, the shift correspond to second channel is
%       reset to zero.
% Output:
%	x:	reconstructed image
%
% Note:	This is the general case of 2D two-channel filterbank
%
% See also:	FBDEC

if ~exist('extmod', 'var')
    extention_mode_string = 'per';
end
if ~exist('intp', 'var')
    flag_interpolation = 0;
end

% Upsampling
switch type1
    case 'q'
        % Quincunx upsampling
        subband_mat1 = quincunx_upsampling(subband_mat1, type2);
        subband_mat2 = quincunx_upsampling(subband_mat2, type2);

    case 'p'
        % Parallelogram upsampling
        subband_mat1 = parallelogram_upsampling(subband_mat1, type2);
        subband_mat2 = parallelogram_upsampling(subband_mat2, type2);

    case 'pq'
        % Quincux upsampling using the equivalent type
        pqtype = {'1r', '2r', '2c', '1c'};

        subband_mat1 = quincunx_upsampling(subband_mat1, pqtype{type2});
        subband_mat2 = quincunx_upsampling(subband_mat2, pqtype{type2});

    otherwise
        error('Invalid input type1');
end

% Stagger sampling if filter is odd-size
if all(mod(size(reconstruction_filter2), 2))
    shift = [1; 0];

    % Account for the resampling matrix in the parallegoram case
    if type1 == 'p'
	    R = {[1, 1; 0, 1], [1, -1; 0, 1], [1, 0; 1, 1], [1, 0; -1, 1]};
	    shift = R{type2} * shift;
    end
else
    shift = [0; 0];
end

if flag_interpolation == 1
    shift = [0; 0]; % reset all delay when doing interpolation
end

% Dimension that has even size filter needs to be adjusted to obtain 
% perfect reconstruction with zero shift
adjust0 = mod(size(reconstruction_filter1) + 1, 2)';
adjust1 = mod(size(reconstruction_filter2) + 1, 2)';

% Extend, filter and keep the original size
x0 = filter_2D_with_edge_handling(subband_mat1, reconstruction_filter1, extention_mode_string, adjust0);
x1 = filter_2D_with_edge_handling(subband_mat2, reconstruction_filter2, extention_mode_string, adjust1 + shift);

% Combine 2 channel to output
x = x0 + x1;

% For parallegoram filterbank using quincunx upsampling,
% a resampling is required at the end
if strcmp(type1,'pq')
    
    % Inverse of resamp(x, type)
    inv_type = [2, 1, 4, 3];
    
    
    x = shift_matrix_columnwise_or_rowwise(x, inv_type(type2));
end