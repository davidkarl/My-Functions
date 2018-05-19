function y = conv_fft_2D(mat_in, filter_mat, shape)
% FCONV2  2-D convolution using fft2
%         Compatible with conv2
%
%	y = fconv2(x, f, shape)
%
% Input:
%   x:      input image
%   shiftsize:  is a vector of integer scalars where
%       the N-th element specifies the shift amount for the N-th dimension
%   shape   : specifies an alternative interpolation method: 
%       'full' 
%       'same'  
%       'valid'  
%
% Output:
%   y:	    subband images 
%
% Note:
%         
% See also: CONV2
% 

if ~exist('shape', 'var')
    shape = 'full';
end

% take size x f
[mat_in_rows,mat_in_columns] = size(mat_in);
[filter_in_rows,filter_in_columns] = size(filter_mat);

% if any(size(x) < size(f))
%     error('Size of image must be larger than filter');
% end

%number of rows and columns to use in fft (power of 2):
M = 2^nextpow2(mat_in_rows+filter_in_rows-1); 
N = 2^nextpow2(mat_in_columns+filter_in_columns-1);

%Perform covolution in fft domain:
y = ifft2(fft2(mat_in,M,N).*fft2(filter_mat,M,N));

switch shape
    case {'full'}
        y = y(1:mat_in_rows+filter_in_rows-1,1:mat_in_columns+filter_in_columns-1);
    case {'same'}
        mbh = ceil((filter_in_rows+1)/2); nbh = ceil((filter_in_columns+1)/2);
        y = y(mbh:mbh+mat_in_rows-1,nbh:nbh+mat_in_columns-1);
    case {'valid'}
        y = y(filter_in_rows:mat_in_rows,filter_in_columns:mat_in_columns);
    otherwise
        disp('unvalid shape');
end