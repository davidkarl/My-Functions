function y = parallelogram_downsampling(mat_in, type, flag_polyphase_component)
% PDOWN   Parallelogram Downsampling
%
% 	y = pdown(x, type, [phase])
%
% Input:
%	x:	input image
%	type:	one of {1, 2, 3, 4} for selecting sampling matrices:
%			P1 = [2, 0; 1, 1]
%			P2 = [2, 0; -1, 1]
%			P3 = [1, 1; 0, 2]
%			P4 = [1, -1; 0, 2]
%	phase:	[optional] 0 or 1 for keeping the zero- or one-polyphase
%		component, (default is 0)
%
% Output:
%	y:	parallelogram downsampled image
%
% Note:
%	These sampling matrices appear in the directional filterbank:
%		P1 = R1 * Q1
%		P2 = R2 * Q2
%		P3 = R3 * Q2
%		P4 = R4 * Q1
%	where R's are resampling matrices and Q's are quincunx matrices
%
% See also:	PPDEC

if ~exist('phase', 'var')
    flag_polyphase_component = 0;
end

% Parallelogram polyphase decomposition by simplifying sampling matrices
% using the Smith decomposition of the quincunx matrices

up = 1;
down = 2;
left = 3;
right = 4;
switch type
    case 1	% P1 = R1 * Q1 = D1 * R3
        if flag_polyphase_component == 0
            y = shift_matrix_columnwise_or_rowwise(mat_in(1:2:end, :), left);
        else
            y = shift_matrix_columnwise_or_rowwise(mat_in(2:2:end, [2:end, 1]), left);
        end
        
    case 2	% P2 = R2 * Q2 = D1 * R4
        if flag_polyphase_component == 0
            y = shift_matrix_columnwise_or_rowwise(mat_in(1:2:end, :), right);
        else
            y = shift_matrix_columnwise_or_rowwise(mat_in(2:2:end, :), right);
        end
        
    case 3	% P3 = R3 * Q2 = D2 * R1
        if flag_polyphase_component == 0
            y = shift_matrix_columnwise_or_rowwise(mat_in(:, 1:2:end), up);
        else
            y = shift_matrix_columnwise_or_rowwise(mat_in([2:end, 1], 2:2:end), up);
        end
        
    case 4	% P4 = R4 * Q1 = D2 * R2
        if flag_polyphase_component == 0
            y = shift_matrix_columnwise_or_rowwise(mat_in(:, 1:2:end), down);
        else
            y = shift_matrix_columnwise_or_rowwise(mat_in(:, 2:2:end), down);
        end
        
    otherwise
        error('Invalid argument type');
end
