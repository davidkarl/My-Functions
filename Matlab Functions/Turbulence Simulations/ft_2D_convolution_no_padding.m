function C = ft_2D_convolution_no_padding(A, B, spacing)
N = length(A);
C = ift2(ft2(A, spacing) .* ft2(B, spacing), 1/(N*spacing));

% spacing=L[meters]/N[number of samples] 
% L=total size of reference plane in meters

