function res = make_2D_fractal_pink_noise(mat_size, fractal_dimensions)
% IM = mkFract(SIZE, FRACT_DIM)
%
% Make a matrix of dimensions SIZE (a [Y X] 2-vector, or a scalar)
% containing fractal (pink) noise with power spectral density of the
% form: 1/f^(5-2*FRACT_DIM).  Image variance is normalized to 1.0.
% FRACT_DIM defaults to 1.0

% TODO: Verify that this  matches Mandelbrot defn of fractal dimension.
%       Make this more efficient!


if ~exist('fractal_dimensions','var')
    fractal_dimensions = 1.0;
end

res = randn(mat_size);
fres = fft2(res);
ctr = ceil((mat_size+1)./2);

shaping_filter = ifftshift( make_2D_ramp(mat_size, -(2.5-fractal_dimensions), ctr) );
shaping_filter(1,1) = 1;  %%DC term

fres = shaping_filter .* fres;
fres = ifft2(fres);

if (max(max(abs(imag(fres)))) > 1e-10)
    error('Symmetry error in creating fractal');
else
    res = real(fres);
    res = res / sqrt(var2(res));
end
