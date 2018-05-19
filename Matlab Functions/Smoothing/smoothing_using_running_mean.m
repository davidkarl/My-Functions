function [smth] = smoothing_using_running_mean(x_input_vec,y_input_vec,one_sided_number_of_points_to_average)

% CSRMEANSMTH    Running mean smooth
%
%	yhat = CSRMEANSMTH(X,Y,N) constructs a running mean smoother for the
%	predictor values X and the response values Y. The smoothness of the
%	resulting nonparametric regression is governed by the number of points
%	N. 
%
%   Note that this is a symmetric neighborhood, so the number of points in
%   the neighborhood is actually 2*N + 1.
%
%	INPUTS:		X is a vector of predictor values
%				Y is a vector of response values
%				N is the number of points
%
%   The ouput of this function is a vector YHAT of smoothed values at the
%   observed X values.
%
%   A plot of the data points and smooth is also produced by this function.
%
%   EXAMPLE:
%
%   load vineyard
%   y = totlugcount;
%   x = row;
%   yhat = csrmeansmth(x,y,3);
%
%   See also CSLOESS, CSLOCPOLY, CSLOCLIN, CSNARDWATS, CSLOESSR, CSBINSMTH,
%   CSSPLINESMTH

%   W. L. and A. R. Martinez, 12/14/07
%   Computational Statistics Toolbox, V 2 

number_of_samples = length(x_input_vec);

% Set the number of data points in each window.
% Use symmetric neighborhoods of size 2N+1.

smth = zeros(1,number_of_samples);
smth(1) = mean(y_input_vec(1:1+one_sided_number_of_points_to_average));
smth(end) = mean(y_input_vec(number_of_samples-one_sided_number_of_points_to_average:number_of_samples));
for i = (one_sided_number_of_points_to_average+1):(number_of_samples-one_sided_number_of_points_to_average)
    smth(i) = mean(y_input_vec(i-one_sided_number_of_points_to_average:i+one_sided_number_of_points_to_average));
end

% Find the lower end of the smooth, 
% using as many to the left as possible.
for i = 2:one_sided_number_of_points_to_average
    smth(i) = mean(y_input_vec(1:i+one_sided_number_of_points_to_average));
end

% Find the upper end of the smooth,
% using as many to the right as possible.
for i = (number_of_samples-one_sided_number_of_points_to_average+1):(number_of_samples-1)
    smth(i) = mean(y_input_vec(i-one_sided_number_of_points_to_average:end));
end


figure
plot(x_input_vec,y_input_vec,'o',x_input_vec,smth)
xlabel('X'),ylabel('Y')
