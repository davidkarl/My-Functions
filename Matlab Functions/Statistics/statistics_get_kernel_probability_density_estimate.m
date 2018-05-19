function prob = statistics_get_kernel_probability_density_estimate(points_to_estimate,input_data_points,window_width)
% CSKERNMD	Product kernel.
%
%	PROB = CSKERNMD(X,DATA,H)
%	This returns the product kernel probability density estimate for a d-dimensional 
%	point x based on the data. Here, d >= 2.
%
%	X represents the point(s) in the domain where you want to get the value of the probability.
%	Thus, X is nn x d, where d is the dimensionality.
%	DATA is the n x d data matrix, where each row is an observation.
%	H is an optional vector of window widths, one for each dimension. The default is the
%	value obtained from the Normal Reference Rule.
%
%	See also CSKERN2D

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox


[number_of_samples,number_of_dimensions] = size(input_data_points);
[number_of_samples_to_estimate,number_of_dimensions_of_points_to_estimate] = size(points_to_estimate);
if number_of_dimensions~=number_of_dimensions_of_points_to_estimate
	error('Dimensionality of data must match dimensionality of x.')
	return
end
prob = zeros(1,number_of_samples_to_estimate);
arg = zeros(number_of_samples,number_of_dimensions);

if nargin == 2
	% Get the window widths using Scott's rule.
	tmp = (4/(number_of_samples(number_of_dimensions+2)))^(1/(number_of_dimensions+4));
	window_width = zeros(1,number_of_dimensions);
	s = cov(input_data_points);
	for i = 1:number_of_dimensions
		window_width(i) = s(i,i)*tmp;
	end
end

for i = 1:number_of_samples_to_estimate	% note that const done at end
	for j = 1:number_of_dimensions
		xx = points_to_estimate(i,j)*ones(number_of_samples,1);
		arg(:,j) = ((xx-input_data_points(:,j))/window_width(j)).^2;
	end
	prob(i) = sum(exp(-.5*sum(arg,2)))/(number_of_samples*prod(window_width)*2*pi);
end


