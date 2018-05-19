function y = statistics_get_chi_square_probability_density_at_points(points_vec,nu_degrees_of_freedom_parameter)
% CSCHIP Chi-square probability density function.
%
%   Y = CSCHIP(X,NU) Returns the value of the
%   chi-square probability density function with NU
%   degrees of freedom, at the given values in X.
% 
%   See also CSCHIC, CSCHIRND

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


if nu_degrees_of_freedom_parameter <= 0
   error('The degrees of freedom must be greater than 0')
   return
end
y = gammp(points_vec,nu_degrees_of_freedom_parameter/2,0.5);