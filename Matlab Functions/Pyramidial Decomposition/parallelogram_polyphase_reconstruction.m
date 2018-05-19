function x = parallelogram_polyphase_reconstruction(p0, p1, type)
% PPREC   Parallelogram Polyphase Reconstruction
%
% 	x = pprec(p0, p1, type)
%
% Input:
%	p0, p1:	two parallelogram polyphase components of the image
%	type:	one of {1, 2, 3, 4} for selecting sampling matrices:
%			P1 = [2, 0; 1, 1]
%			P2 = [2, 0; -1, 1]
%			P3 = [1, 1; 0, 2]
%			P4 = [1, -1; 0, 2]
%
% Output:
%	x:	reconstructed image
%
% Note:
%	These sampling matrices appear in the directional filterbank:
%		P1 = R1 * Q1
%		P2 = R2 * Q2
%		P3 = R3 * Q2
%		P4 = R4 * Q1
%	where R's are resampling matrices and Q's are quincunx matrices
%
%	Also note that R1 * R2 = R3 * R4 = I so for example,
%	upsample by R1 is the same with down sample by R2
%
% See also:	PPDEC

% Parallelogram polyphase decomposition by simplifying sampling matrices
% using the Smith decomposition of the quincunx matrices

[m, n] = size(p0);
up = 1;
down = 2;
left = 3;
right = 4;
switch type
    case 1	% P1 = R1 * Q1 = D1 * R3
        x = zeros(2*m, n);
        
        x(1:2:end, :) = shift_matrix_columnwise_or_rowwise(p0, right);
        x(2:2:end, [2:end, 1]) = shift_matrix_columnwise_or_rowwise(p1, right);
        
    case 2	% P2 = R2 * Q2 = D1 * R4
        x = zeros(2*m, n);
        
        x(1:2:end, :) = shift_matrix_columnwise_or_rowwise(p0, left);
        x(2:2:end, :) = shift_matrix_columnwise_or_rowwise(p1, left);
        
    case 3	% P3 = R3 * Q2 = D2 * R1
        x = zeros(m, 2*n);
        
        x(:, 1:2:end) = shift_matrix_columnwise_or_rowwise(p0, down);
        x([2:end, 1], 2:2:end) = shift_matrix_columnwise_or_rowwise(p1, down);
        
    case 4	% P4 = R4 * Q1 = D2 * R2
        x = zeros(m, 2*n);
        
        x(:, 1:2:end) = shift_matrix_columnwise_or_rowwise(p0, up);
        x(:, 2:2:end) = shift_matrix_columnwise_or_rowwise(p1, up);
        
    otherwise
        error('Invalid argument type');
end
