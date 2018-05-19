function y = quincunx_downsampling(mat_in, type, extmod, flag_polyphase_phase)
% QDOWN   Quincunx Downsampling
%
% 	y = qdown(x, [type], [extmod], [phase])
%
% Input:
%	x:	input image
%	type:	[optional] one of {'1r', '1c', '2r', '2c'} (default is '1r')
%		'1' or '2' for selecting the quincunx matrices:
%			Q1 = [1, -1; 1, 1] or Q2 = [1, 1; -1, 1]
%		'r' or 'c' for suppresing row or column
%	phase:	[optional] 0 or 1 for keeping the zero- or one-polyphase
%		component, (default is 0)
%
% Output:
%	y:	qunincunx downsampled image
%
% See also:	QPDEC

if ~exist('type', 'var')
    type = '1r';
end

if ~exist('phase', 'var')
    flag_polyphase_phase = 0;
end

% Quincunx downsampling using the Smith decomposition:
%	Q1 = R2 * [2, 0; 0, 1] * R3
%	   = R3 * [1, 0; 0, 2] * R2
% and,
%	Q2 = R1 * [2, 0; 0, 1] * R4
%	   = R4 * [1, 0; 0, 2] * R1
%
% See RESAMP for the definition of those resampling matrices
up = 1;
down = 2;
left = 3;
right = 4;
switch type
    case {'1r'}
        z = shift_matrix_columnwise_or_rowwise(mat_in, down);
        
        if flag_polyphase_phase == 0
            y = shift_matrix_columnwise_or_rowwise(z(1:2:end, :), left);
        else
            y = shift_matrix_columnwise_or_rowwise(z(2:2:end, [2:end, 1]), left);
        end
        
    case {'1c'}
        z = shift_matrix_columnwise_or_rowwise(mat_in, left);
        
        if flag_polyphase_phase == 0
            y = shift_matrix_columnwise_or_rowwise(z(:, 1:2:end), down);
        else
            y = shift_matrix_columnwise_or_rowwise(z(:, 2:2:end), down);
        end
        
    case {'2r'}
        z = shift_matrix_columnwise_or_rowwise(mat_in, up);
        
        if flag_polyphase_phase == 0
            y = shift_matrix_columnwise_or_rowwise(z(1:2:end, :), right);
        else
            y = shift_matrix_columnwise_or_rowwise(z(2:2:end, :), right);
        end
        
    case {'2c'}
        z = shift_matrix_columnwise_or_rowwise(mat_in, right);
        
        if flag_polyphase_phase == 0
            y = shift_matrix_columnwise_or_rowwise(z(:, 1:2:end), up);
        else
            y = shift_matrix_columnwise_or_rowwise(z([2:end, 1], 2:2:end), up);
        end
        
    otherwise
        error('Invalid argument type');
end
