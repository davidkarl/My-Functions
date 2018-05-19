function [p0, p1] = quincunx_polyphase_decomposition(mat_in, type)
% QPDEC   Quincunx Polyphase Decomposition
%
% 	[p0, p1] = qpdec(x, [type])
%
% Input:
%	x:	input image
%	type:	[optional] one of {'1r', '1c', '2r', '2c'} default is '1r'
%		'1' and '2' for selecting the quincunx matrices:
%			Q1 = [1, -1; 1, 1] or Q2 = [1, 1; -1, 1]
%		'r' and 'c' for suppresing row or column
%
% Output:
%	p0, p1:	two qunincunx polyphase components of the image

if ~exist('type', 'var')
    type = '1r';
end

% Quincunx downsampling using the Smith decomposition:
%	Q1 = R2 * D1 * R3
%	   = R3 * D2 * R2
% and,
%	Q2 = R1 * D1 * R4
%	   = R4 * D2 * R1
%
% where D1 = [2, 0; 0, 1] and D2 = [1, 0; 0, 2].
% See RESAMP for the definition of the resampling matrices R's

up = 1;
down = 2;
left = 3;
right = 4;
switch type
    case {'1r'}		% Q1 = R2 * D1 * R3
        y = shift_matrix_columnwise_or_rowwise(mat_in, down);
        
        p0 = shift_matrix_columnwise_or_rowwise(y(1:2:end, :), left);
        
        % inv(R2) * [0; 1] = [1; 1]
        p1 = shift_matrix_columnwise_or_rowwise(y(2:2:end, [2:end, 1]), left);
        
    case {'1c'}		% Q1 = R3 * D2 * R2
        y = shift_matrix_columnwise_or_rowwise(mat_in, left);
        
        p0 = shift_matrix_columnwise_or_rowwise(y(:, 1:2:end), down);
        
        % inv(R3) * [0; 1] = [0; 1]
        p1 = shift_matrix_columnwise_or_rowwise(y(:, 2:2:end), down);
        
    case {'2r'}		% Q2 = R1 * D1 * R4
        y = shift_matrix_columnwise_or_rowwise(mat_in, up);
        
        p0 = shift_matrix_columnwise_or_rowwise(y(1:2:end, :), right);
        
        % inv(R1) * [1; 0] = [1; 0]
        p1 = shift_matrix_columnwise_or_rowwise(y(2:2:end, :), right);
        
    case {'2c'}		% Q2 = R4 * D2 * R1
        y = shift_matrix_columnwise_or_rowwise(mat_in, right);
        
        p0 = shift_matrix_columnwise_or_rowwise(y(:, 1:2:end), up);
        
        % inv(R4) * [1; 0] = [1; 1]
        p1 = shift_matrix_columnwise_or_rowwise(y([2:end, 1], 2:2:end), up);
        
    otherwise
        error('Invalid argument type');
end
