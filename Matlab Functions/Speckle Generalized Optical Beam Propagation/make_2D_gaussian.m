function [res] = make_2D_gaussian(mat_size, covariance_mat, gaussian_center, gaussian_amplitude)
% IM = mkGaussian(SIZE, COVARIANCE, MEAN, AMPLITUDE)
%
% Compute a matrix with dimensions SIZE (a [Y X] 2-vector, or a
% scalar) containing a Gaussian function, centered at pixel position
% specified by MEAN (default = (size+1)/2), with given COVARIANCE (can
% be a scalar, 2-vector, or 2x2 matrix.  Default = (min(size)/6)^2),
% and AMPLITUDE.  AMPLITUDE='norm' (default) will produce a
% probability-normalized function.  All but the first argument are
% optional.


mat_size = mat_size(:);
if (size(mat_size,1) == 1)
    mat_size = [mat_size,mat_size];
end

%------------------------------------------------------------
% OPTIONAL ARGS:

if ~exist('covariance_mat','var')
    covariance_mat = (min(mat_size(1),mat_size(2))/6)^2;
end

if ( ~exist('gaussian_mean','var') || isempty(gaussian_center) )
    gaussian_center = (mat_size+1)/2;
else
    gaussian_center = gaussian_center(:);
    if (size(gaussian_center,1) == 1)
        gaussian_center = [gaussian_center, gaussian_center];
    end
end

if ~exist('gaussian_amplitude','var')
    gaussian_amplitude = 'norm';
end

%------------------------------------------------------------

[xramp,yramp] = meshgrid([1:mat_size(2)]-gaussian_center(2),[1:mat_size(1)]-gaussian_center(1));

if (sum(size(covariance_mat)) == 2)  % scalar
    if (strcmp(gaussian_amplitude,'norm'))
        gaussian_amplitude = 1/(2*pi*covariance_mat(1));
    end
    exponential_argument = (xramp.^2 + yramp.^2)/(-2 * covariance_mat);
elseif (sum(size(covariance_mat)) == 3) % a 2-vector
    if (strcmp(gaussian_amplitude,'norm'))
        gaussian_amplitude = 1/(2*pi*sqrt(covariance_mat(1)*covariance_mat(2)));
    end
    exponential_argument = xramp.^2/(-2 * covariance_mat(2)) + yramp.^2/(-2 * covariance_mat(1));
else
    %if covariance matrix is actually a matrix then we need to calculate the determinant:
    if (strcmp(gaussian_amplitude,'norm'))
        gaussian_amplitude = 1/(2*pi*sqrt(det(covariance_mat)));
    end
    covariance_mat = -inv(covariance_mat)/2;
    exponential_argument = covariance_mat(2,2)*xramp.^2 + (covariance_mat(1,2)+covariance_mat(2,1))*(xramp.*yramp) ...
        + covariance_mat(1,1)*yramp.^2;
end

res = gaussian_amplitude .* exp(exponential_argument);
