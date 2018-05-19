function C = ft_1D_convolution_no_padding(A, B, spacing)
N = length(A);
C = ift(ft(A, spacing) .* ft(B, spacing), 1/(N*spacing));

% spacing=L[meters]/N[number of samples] 
% L=total size of reference plane in meters

