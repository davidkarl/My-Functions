function y = upsample_specify_each_dimension(mat_in, upsampling_factors_vec, polyphase_components_vec)
% DUP   Diagonal Upsampling
%
%	y = dup(x, step, [phase])
%
% Input:
%	x:	input image
%	step:	upsampling factors for each dimension which should be a
%		2-vector
%	phase:	[optional] to specify the phase of the input image which
%		should be less than step, (default is [0, 0])
%		If phase == 'minimum', a minimum size of upsampled image
%		is returned
%
% Output:
%	y:	diagonal upsampled image
%
% See also:	DDOWN

if ~exist('phase', 'var')
    polyphase_components_vec = [0, 0];
end

mat_in_size = size(mat_in);

if lower(polyphase_components_vec(1)) == 'm'
    y = zeros((mat_in_size - 1) .* upsampling_factors_vec + 1);
    y(1:upsampling_factors_vec(1):end, 1:upsampling_factors_vec(2):end) = mat_in;
    
else
    y = zeros(mat_in_size .* upsampling_factors_vec);
    y(1+polyphase_components_vec(1) : upsampling_factors_vec(1) : end, ...
      1+polyphase_components_vec(2) : upsampling_factors_vec(2) : end) = mat_in;
end
