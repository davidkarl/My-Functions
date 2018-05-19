function [y0, y1] = two_channel_2D_filterbank_decomposition(mat_in, ...
    filter_mat1, ...
    filter_mat2, ...
    downsampling_matrix_type, ...
    downsampling_matrix_parameter, ...
    extention_mode_string)
% FBDEC   Two-channel 2D Filterbank Decomposition
%
%	[y0, y1] = fbdec(x, h0, h1, type1, type2, [extmod])
%
% Input:
%	x:	input image
%	h0, h1:	two decomposition 2D filters
%	type1:	'q', 'p' or 'pq' for selecting quincunx or parallelogram
%		downsampling matrix
%	type2:	second parameter for selecting the filterbank type
%		If type1 == 'q' then type2 is one of {'1r', '1c', '2r', '2c'}
%		If type1 == 'p' then type2 is one of {1, 2, 3, 4}
%			Those are specified in QDOWN and PDOWN
%		If type1 == 'pq' then same as 'p' except that
%		the paralellogram matrix is replaced by a combination
%		of a  resampling and a quincunx matrices
%	extmod:	[optional] extension mode (default is 'per')
%
% Output:
%	y0, y1:	two result subband images
%
% Note:		This is the general implementation of 2D two-channel
% 		filterbank
%
% See also:	FBDEC_SP

if ~exist('extmod', 'var')
    extention_mode_string = 'per';
end

%For parallegoram filterbank using quincunx downsampling, resampling is applied before filtering
if strcmp(downsampling_matrix_type,'pq')
    mat_in = shift_matrix_columnwise_or_rowwise(mat_in, downsampling_matrix_parameter);
end

%Stagger sampling if filter is odd-size (in both dimensions):
if all(mod(size(filter_mat2), 2))
    shift = [-1 ; 0];
    
    %Account for the resampling matrix in the parallegoram case
    if downsampling_matrix_type == 'p'
        R = {[1, 1; 0, 1], [1, -1; 0, 1], [1, 0; 1, 1], [1, 0; -1, 1]};
        shift = R{downsampling_matrix_parameter} * shift; %????
    end
    
else
    shift = [0; 0];
end

%Extend, filter and keep the original size:
y0 = filter_2D_with_edge_handling(mat_in, filter_mat1, extention_mode_string);
y1 = filter_2D_with_edge_handling(mat_in, filter_mat2, extention_mode_string, shift);

%Downsampling:
switch downsampling_matrix_type
    case 'q'
        % Quincunx downsampling
        y0 = quincunx_downsampling(y0, downsampling_matrix_parameter);
        y1 = quincunx_downsampling(y1, downsampling_matrix_parameter);
        
    case 'p'
        % Parallelogram downsampling
        y0 = parallelogram_downsampling(y0, downsampling_matrix_parameter);
        y1 = parallelogram_downsampling(y1, downsampling_matrix_parameter);
        
    case 'pq'
        % Quincux downsampling using the equipvalent type
        pqtype = {'1r', '2r', '2c', '1c'};
        
        y0 = quincunx_downsampling(y0, pqtype{downsampling_matrix_parameter});
        y1 = quincunx_downsampling(y1, pqtype{downsampling_matrix_parameter});
        
    otherwise
        error('Invalid input type1');
end