function yhat = statistics_get_values_using_loess_smoothing(x_input_vec,y_input_vec,points_to_estimate_vec,...
                                                            smoothing_parameter,local_fit_degree)
% CSLOESS   Basic loess smoothing
%
%   YHAT = CSLOESS(X,Y,XO,ALPHA,DEG)
%
%   This function performs the basic loess smoothing for univariate data.
%   YHAT is the value of the smooth. X and Y are the observed data. XO
%   is the domain over which to evaluate the smooth YHAT. ALPHA is the 
%   smoothing parameter, and DEG is the degree of the local fit (1 or 2).
%
%
%   EXAMPLE:
%    
%   load salmon
%   x = salmon(:,1);
%   y = salmon(:,2);
%   [xs,inds] = sort(x);
%   ys = y(inds);
%   yhat = csloess(xs,ys,xs,0.5,2);
%   plot(xs,ys,'.',xs,yhat)
%
%   See also CSLOCPOLY, CSLOCLIN, CSNARDWATS, CSLOESSR, CSRMEANSMTH, 
%   CSBINSMTH, CSSPLINESMTH


%   W. L. and A. R. Martinez, May 2007
%   CS Toolbox


if local_fit_degree ~= 1 && local_fit_degree ~= 2
	error('Degree of local fit must be 1 or 2')
end
if smoothing_parameter <= 0 || smoothing_parameter >= 1
	error('Alpha must be between 0 and 1')
end
if length(x_input_vec) ~= length(y_input_vec)
	error('Input vectors do not have the same length.')
end

%get constants needed:
number_of_input_samples = length(x_input_vec);
number_of_neighbouring_samples_to_use = floor(smoothing_parameter*number_of_input_samples);

%set up the memory:
yhat = zeros(size(points_to_estimate_vec));

%for each xo, find the k points that are closest:
for i = 1:length(points_to_estimate_vec)
	distances_to_other_points_vec = abs(points_to_estimate_vec(i) - x_input_vec);
	[sorted_distances_vec,ind] = sort(distances_to_other_points_vec);
	relevant_x_points = x_input_vec(ind(1:number_of_neighbouring_samples_to_use));	% get the points in the neighborhood
	relevant_y_points = y_input_vec(ind(1:number_of_neighbouring_samples_to_use));
	max_relevant_distances_point = sorted_distances_vec(number_of_neighbouring_samples_to_use);  %% Check this
	sorted_distances_vec((number_of_neighbouring_samples_to_use+1):number_of_input_samples) = [];
	u = sorted_distances_vec/max_relevant_distances_point;
	weights_for_least_squares = (1 - u.^3).^3;
	p = weighted_least_squares_fit(relevant_x_points,relevant_y_points,weights_for_least_squares,local_fit_degree);
	yhat(i) = polyval(p,points_to_estimate_vec(i));
end

function p = weighted_least_squares_fit(x,y,w,deg)
% This will perform the weighted least squares
n = length(x);
x = x(:);
y = y(:);
w = w(:);
% get matrices
W = spdiags(w,0,n,n);
A = vander(x); %   A = VANDER(V) returns the Vandermonde matrix whose columns
               %   are powers of the vector V, that is A(i,j) = v(i)^(n-j).

A(:,1:length(x)-deg-1) = [];
V = A'*W*A;
Y = A'*W*y;
[Q,R] = qr(V,0); 
p = R\(Q'*Y); 
p = p';		% to fit MATLAB convention
