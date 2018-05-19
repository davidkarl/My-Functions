function reconstructed_image = quincunx_polyphase_reconstruction(p0, p1, type)
% QPREC   Quincunx Polyphase Reconstruction
%
% 	x = qprec(p0, p1, [type])
%
% Input:
%	p0, p1:	two qunincunx polyphase components of the image
%	type:	[optional] one of {'1r', '1c', '2r', '2c'}, default is '1r'
%		'1' and '2' for selecting the quincunx matrices:
%			Q1 = [1, -1; 1, 1] or Q2 = [1, 1; -1, 1]
%		'r' and 'c' for suppresing row or column
%
% Output:
%	x:	reconstructed image
%
% Note:
%	Note that R1 * R2 = R3 * R4 = I so for example,
%	upsample by R1 is the same with down sample by R2
%
% See also:	QPDEC

if ~exist('type', 'var')
    type = '1r';
end

% Quincunx downsampling using the Smith decomposition:
%
%       Q1 = R2 * D1 * R3
%          = R3 * D2 * R2
% and,
%       Q2 = R1 * D1 * R4
%          = R4 * D2 * R1
%
% where D1 = [2, 0; 0, 1] and D2 = [1, 0; 0, 2].
% See RESAMP for the definition of the resampling matrices R's

[m, n] = size(p0);
up = 1;
down = 2;
left = 3;
right = 4;
switch type
    case {'1r'}		% Q1 = R2 * D1 * R3
        y = zeros(2*m, n);
        
        y(1:2:end, :) = shift_matrix_columnwise_or_rowwise(p0, right);
        y(2:2:end, [2:end, 1]) = shift_matrix_columnwise_or_rowwise(p1, right);
        
        reconstructed_image = shift_matrix_columnwise_or_rowwise(y, up);
        
    case {'1c'}		% Q1 = R3 * D2 * R2
        y = zeros(m, 2*n);
        
        y(:, 1:2:end) = shift_matrix_columnwise_or_rowwise(p0, up);
        y(:, 2:2:end) = shift_matrix_columnwise_or_rowwise(p1, up);
        
        reconstructed_image = shift_matrix_columnwise_or_rowwise(y, right);
        
    case {'2r'}		% Q2 = R1 * D1 * R4
        y = zeros(2*m, n);
        
        y(1:2:end, :) = shift_matrix_columnwise_or_rowwise(p0, left);
        y(2:2:end, :) = shift_matrix_columnwise_or_rowwise(p1, left);
        
        reconstructed_image = shift_matrix_columnwise_or_rowwise(y, down);
        
    case {'2c'}		% Q2 = R4 * D2 * R1
        y = zeros(m, 2*n);
        
        y(:, 1:2:end) = shift_matrix_columnwise_or_rowwise(p0, down);
        y([2:end, 1], 2:2:end) = shift_matrix_columnwise_or_rowwise(p1, down);
        
        reconstructed_image = shift_matrix_columnwise_or_rowwise(y, left);
        
    otherwise
        error('Invalid argument type');
end
