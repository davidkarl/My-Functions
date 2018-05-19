function [fhat,window_width] = statistics_get_kernel_density_estimator_1D(points_to_estimate,input_data_vec,window_width)
% CSKERN1D  Univariate kernel density estimator.
%
%   [FHAT,h] = CSKERN1D(X,DATA,H)
%
%   This function returns the univariate kernel density
%   estimate using the observations in DATA. The domain values
%   over which to evaluate the density are given in X.
%
%   The smoothing parameter is given by the optional window
%   width H. The default value is obtained from the Normal
%   Reference Rule.
%
%   The window width H is returned by the function, so one can
%   adjust the value.
%
%   The normal kernel is used.
%
%   EXAMPLE:
%
%   load snowfall
%   x = linspace(10, 150);
%   [fhat,h] = cskern1d(x,snowfall,4);
%   
%
%   See also CSHISTDEN, CSFREQPOLY, CSASH

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

number_of_samples = length(input_data_vec);
fhat = zeros(size(points_to_estimate));

if nargin ==2
    window_width = 1.06*number_of_samples^(-1/5);
end
for i=1:number_of_samples
   % get each kernel function evaluated at x
			% centered 		at data
   f = exp(-(1/(2*window_width^2))*(points_to_estimate-input_data_vec(i)).^2)/sqrt(2*pi)/window_width;
   fhat = fhat+f/(number_of_samples);
end

plot(points_to_estimate,fhat)




