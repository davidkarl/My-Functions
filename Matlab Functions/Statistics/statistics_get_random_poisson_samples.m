function x = statistics_get_random_poisson_samples(lambda_parameter,number_of_samples)
% CSPOIRND Generate Poisson random variables.
%
%   R = CSPOIRND(LAMBDA, N) Generates N random variables
%   from the Poisson distribution with parameter LAMBDA.
%   R is a row vector.
%
%   See also CSPOISC, CSPOIRND, CSPOIPAR, CSPOISSPLOT


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 
%   The reference for this is Ross, 1997, page 50.


x = zeros(1,number_of_samples);
j = 1;
while j <= number_of_samples
   flag = 1;
   % initialize quantities
   u = rand(1);
   i = 0;
   p = exp(-lambda_parameter);
   F = p;
   while flag	% generate the variate needed
      if u <= F % then accept
         x(j) = i;
         flag = 0;
         j=j+1;
      else 	% move to next prob'y
         p = lambda_parameter*p/(i+1);
         i = i+1;
         F = F + p;
      end
   end
end
