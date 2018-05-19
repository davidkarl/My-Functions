function yhat = statistics_get_values_using_local_polynomial_kernel_estimator(...
                   x_input_points,y_input_points,points_to_estimate_vec,polynomial_degree_to_use,window_width)
% CSLOCPOLY     Local polynomial kernel estimator.
%
%   YHAT = CSLOCPOLY(X,Y,XO,DEG,H)
%
%   This performs nomparametric regression using the local
%   polynomial kernel estimator. The observed data are given
%   in X and Y. The domain over which to evaluate the function
%   is given in the vector XO. The degree of the local fit it
%   DEG, and the smoothing is governed by the bandwidth of 
%   the kernel, H.
%
%   EXAMPLE:
%   
%   load salmon
%   x = salmon(:,1);
%   y = salmon(:,2);
%   [xs,inds] = sort(x);
%   ys = y(inds);
%   yhat = cslocpoly(xs,ys,xs,1,75);
%   plot(xs,ys,'.',xs,yhat)
%
%   See also CSLOESS, CSLOCLIN, CSNARDWATS, CSBINSMTH, CSRMEANSMTH,
%   CSSPLINESMTH

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox


% we will use the normal for the kernel and to get the weights
% This means that all points get weighted.
mystrg = '(2*pi*h^2)^(-1/2)*exp(-0.5*((x - mu)/h).^2)';
wfun = inline(mystrg);

number_of_input_samples = length(x_input_points);
number_of_points_to_estimate = length(points_to_estimate_vec);
yhat = zeros(size(points_to_estimate_vec));
x_input_points = x_input_points(:);	% so it will be dimensionally compliant
y_input_points = y_input_points(:);

% get the fit at all points in xo
for i = 1:number_of_points_to_estimate
	% center the x values at xo
	xc = x_input_points - points_to_estimate_vec(i);
	weights_vec = wfun(window_width,points_to_estimate_vec(i),x_input_points);	
	% note that xo(i) is the mean, 
	% the kernel is evaluated at all points in x
	W = diag(weights_vec);
	A = vander(xc);
	A(:,1:(number_of_input_samples-polynomial_degree_to_use-1)) = [];
	V = A'*W*A;
	Y = A'*W*y_input_points;
	[Q,R] = qr(V,0); 
	p = R\(Q'*Y); 
	p = p';
	yhat(i) = p(end);
end
