function X = statistics_get_random_chi_square_vec(number_of_samples,nu_degrees_of_freedom_parameter)
% CSCHIRND Chi-square random variates.
%
%   X = CSCHIRND(N,MU) Returns an array of N Chi-square
%   random variables with degrees of freedom NU.
%
%   See also CSCHISP, CSCHISC

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 

% generate the uniforms needed depending on whether the 
rm = rem(nu_degrees_of_freedom_parameter,2);
k = floor(nu_degrees_of_freedom_parameter/2);
if rm == 0	% then even degrees of freedom
   U = rand(k,number_of_samples);
   if k ~= 1
      X = -2*log(prod(U));
   else
      X = -2*log(U);
   end
else
   U = rand(k,number_of_samples);
   Z = randn(1,number_of_samples);
   if k ~= 1
      X = Z.^2-2*log(prod(U));
   else
      X = Z.^2-2*log(U);
   end
end

