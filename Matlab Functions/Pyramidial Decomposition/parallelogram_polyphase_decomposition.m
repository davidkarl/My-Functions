function [p0, p1] = parallelogram_polyphase_decomposition(mat_in, type)
% PPDEC   Parallelogram Polyphase Decomposition
%
% 	[p0, p1] = ppdec(x, type)
%
% Input:
%	x:	input image
%	type:	one of {1, 2, 3, 4} for selecting sampling matrices:
%			P1 = [2, 0; 1, 1]
%			P2 = [2, 0; -1, 1]
%			P3 = [1, 1; 0, 2]
%			P4 = [1, -1; 0, 2]
%
% Output:
%	p0, p1:	two parallelogram polyphase components of the image
%
% Note:
%	These sampling matrices appear in the directional filterbank:
%		P1 = R1 * Q1
%		P2 = R2 * Q2
%		P3 = R3 * Q2
%		P4 = R4 * Q1
%	where R's are resampling matrices and Q's are quincunx matrices
%
% See also:	QPDEC

% Parallelogram polyphase decomposition by simplifying sampling matrices
% using the Smith decomposition of the quincunx matrices

up = 1;
down = 2;
left = 3;
right = 4;
switch type
    case 1	% P1 = R1 * Q1 = D1 * R3
        p0 = shift_matrix_columnwise_or_rowwise(mat_in(1:2:end, :), left);
        
        % R1 * [0; 1] = [1; 1]
        p1 = shift_matrix_columnwise_or_rowwise(mat_in(2:2:end, [2:end, 1]), left);
        
    case 2	% P2 = R2 * Q2 = D1 * R4
        p0 = shift_matrix_columnwise_or_rowwise(mat_in(1:2:end, :), right);
        
        % R2 * [1; 0] = [1; 0]
        p1 = shift_matrix_columnwise_or_rowwise(mat_in(2:2:end, :), right);
        
    case 3	% P3 = R3 * Q2 = D2 * R1
        p0 = shift_matrix_columnwise_or_rowwise(mat_in(:, 1:2:end), up);
        
        % R3 * [1; 0] = [1; 1]
        p1 = shift_matrix_columnwise_or_rowwise(mat_in([2:end, 1], 2:2:end), up);
        
    case 4	% P4 = R4 * Q1 = D2 * R2
        p0 = shift_matrix_columnwise_or_rowwise(mat_in(:, 1:2:end), down);
        
        % R4 * [0; 1] = [0; 1]
        p1 = shift_matrix_columnwise_or_rowwise(mat_in(:, 2:2:end), down);
        
    otherwise
        error('Invalid argument type');
end
