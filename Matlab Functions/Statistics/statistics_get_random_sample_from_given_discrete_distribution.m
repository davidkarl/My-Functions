function rs = statistics_get_random_sample_from_given_discrete_distribution(x_values,p_values,number_of_samples_to_generate)
% CSSAMPLE Random sample from an arbitrary discrete/finite distribution.
%
%	R = CSSAMPLE(X,P,N) This function will take an arbitrary discrete.
%	finite distribution and return a random sample from it. 
%	The domain of the function is X. These are the values that the random 
%	variable can assume. The probability associated with each one is given in the
% 	vector P. The number of variates that will be generated is N.

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

if length(x_values) ~= length(p_values)
   error('The size of the input vectors do not match.')
   return
end

%sort just in case they are not in order:
[xs,ind] =s ort(x_values);
ps = p_values(ind);	% sort these in the same order as x

%Get the cdf:
F = cumsum(ps);

%Find all of the required variates:
for i=1:number_of_samples_to_generate
   u = rand(1,1);
   if u<= F(1)
      rs(i) = x_values(1);
   elseif u > F(end-1)
      rs(i) = x_values(end);
   else
      ind = find(u <= F);
      rs(i) = xs(ind(1));
   end
end





