function [y0, y1] = two_channel_2D_filterbank_decomposition_ladder_structure(mat_in, ...
                                                                             filter_ladder_network, ...
                                                                             downsampling_matrix_type, ...
                                                                             downsampling_parameter, ...
                                                                             extention_mode_string)
% FBDEC_L   Two-channel 2D Filterbank Decomposition using Ladder Structure
%
%	[y0, y1] = fbdec_l(x, f, type1, type2, [extmod])
%
% Input:
%	x:	input image
%	f:	filter in the ladder network structure
%	type1:	'q' or 'p' for selecting quincunx or parallelogram
%		downsampling matrix
%	type2:	second parameter for selecting the filterbank type
%		If type1 == 'q' then type2 is one of {'1r', '1c', '2r', '2c'}
%			({2, 3, 1, 4} can also be used as equivalent)
%		If type1 == 'p' then type2 is one of {1, 2, 3, 4}
%		Those are specified in QPDEC and PPDEC
%	extmod:	[optional] extension mode (default is 'per')
%		This refers to polyphase components.
%
% Output:
%	y0, y1:	two result subband images
%
% Note:		This is also called the lifting scheme
%
% See also:	FBDEC, FBREC_L

% Modulate f
filter_ladder_network(1:2:end) = -filter_ladder_network(1:2:end);

if min(size(mat_in)) == 1
    error('Input is a vector, unpredicted output!');
end

if ~exist('extmod', 'var')
    extention_mode_string = 'per';
end

% Polyphase decomposition of the input image
switch lower(downsampling_matrix_type(1))
    case 'q'
        % Quincunx polyphase decomposition
	    [p0, p1] = quincunx_polyphase_decomposition(mat_in, downsampling_parameter);
	
    case 'p'
	    % Parallelogram polyphase decomposition
	    [p0, p1] = quincunx_polyphase_decomposition(mat_in, downsampling_parameter);
	
    otherwise
	    error('Invalid argument type1');
end

%Ladder network structure:
y0 = (1 / sqrt(2)) * ...
    (p0 - filter_2D_seperable_filtering_with_extension_handling(p1, filter_ladder_network, filter_ladder_network, extention_mode_string, [1, 1]));
y1 = (sqrt(2) * p1) + ...
    filter_2D_seperable_filtering_with_extension_handling(y0, filter_ladder_network, filter_ladder_network, extention_mode_string);




