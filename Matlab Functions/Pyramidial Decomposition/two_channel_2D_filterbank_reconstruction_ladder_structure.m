function x = two_channel_2D_filterbank_reconstruction_ladder_structure(subband_mat1, ...
                                                                       subband_mat2, ...
                                                                       filter_ladder_network, ...
                                                                       downsampling_matrix_type, ...
                                                                       downsampling_matrix_parameter, ...
                                                                       extention_mode_string)
% FBREC_L   Two-channel 2D Filterbank Reconstruction using Ladder Structure 
%
%	x = fbrec_l(y0, y1, f, type1, type2, [extmod])
%
% Input:
%	y0, y1:	two input subband images
%	f:	filter in the ladder network structure
%	type1:	'q' or 'p' for selecting quincunx or parallelogram
%		downsampling matrix
%	type2:	second parameter for selecting the filterbank type
%		If type1 == 'q' then type2 is one of {'1r', '1c', '2r', '2c'}
%			({2, 3, 1, 4} can also be used as equivalent)
%		If type1 == 'p' then type2 is one of {1, 2, 3, 4}
%		Those are specified in QPDEC and PPDEC
%       extmod: [optional] extension mode (default is 'per')
%		This refers to polyphase components.
%
% Output:
%	x:	reconstructed image
%
% Note:		This is also called the lifting scheme	
%
% See also:	FBDEC_L

% Modulate f
filter_ladder_network(1:2:end) = -filter_ladder_network(1:2:end);

if ~exist('extmod', 'var')
    extention_mode_string = 'per';
end

% Ladder network structure
p1 = (-1 / sqrt(2)) * ...
    (-subband_mat2 + filter_2D_seperable_filtering_with_extension_handling(subband_mat1, filter_ladder_network, filter_ladder_network, extention_mode_string));

p0 = sqrt(2) * subband_mat1 + ...
    filter_2D_seperable_filtering_with_extension_handling(p1, filter_ladder_network, filter_ladder_network, extention_mode_string, [1, 1]);

% Polyphase reconstruction
switch lower(downsampling_matrix_type(1))
    case 'q'
        % Quincunx polyphase reconstruction
        x = quincunx_polyphase_reconstruction(p0, p1, downsampling_matrix_parameter);
	
    case 'p'
        % Parallelogram polyphase reconstruction
        x = quincunx_polyphase_reconstruction(p0, p1, downsampling_matrix_parameter);
        
    otherwise
        error('Invalid argument type1');
end