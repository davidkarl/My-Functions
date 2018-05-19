function yhat = statistics_get_values_using_robust_loess_smoothing(...
                                x_input_points,y_input_points,points_to_estimate,alpha_smoothing_parameter,fit_degree)
% CSLOESSR  Robust loess smoothing.
%
%   YHAT = CSLOESSR(X,Y,XO,ALPHA,DEG)
%
%   This function performs the robust loess smoothing for univariate data.
%   YHAT is the value of the smooth. X and Y are the observed data. XO
%   is the domain over which to evaluate the smooth YHAT. ALPHA is the 
%   smoothing parameter, and DEG is the degree of the local fit (1 or 2).
%
%   EXAMPLE:
%   
%   load salmon
%   x = salmon(:,1);
%   y = salmon(:,2);
%   [xs,inds] = sort(x);
%   ys = y(inds);
%   yhat = csloessr(xs,ys,xs,0.5,2);
%   plot(xs,ys,'.',xs,yhat)
%
%   See also CSLOCPOLY, CSLOCLIN, CSNARDWATS, CSLOESS, CSRMEANSMTH,
%   CSBINSMTH, CSSPLINESMTH


%   W. L. and A. R. Martinez, 3-4-04


if fit_degree ~= 1 && fit_degree ~= 2
	error('Degree of local fit must be 1 or 2')
end
if alpha_smoothing_parameter <= 0 || alpha_smoothing_parameter >= 1
	error('Alpha must be between 0 and 1')
end
if length(x_input_points) ~= length(y_input_points)
	error('Input vectors do not have the same length.')
end

%get constants needed:
number_of_samples = length(x_input_points);
number_of_neighbouring_samples = floor(alpha_smoothing_parameter*number_of_samples);
convergence_tolerance = 0.003;	% convergence tolerance for robust procedure
max_number_of_iterations = 50;	% maximum allowed number of iterations

%set up the memory:
yhat = zeros(size(points_to_estimate));

%for each xo, find the k points that are closest First find the initial loess fit:
for i = 1:length(points_to_estimate)
	dist = abs(points_to_estimate(i) - x_input_points);
	[sdist,ind] = sort(dist);
	Nxo = x_input_points(ind(1:number_of_neighbouring_samples));	% get the points in the neighborhood
	Nyo = y_input_points(ind(1:number_of_neighbouring_samples));
	delxo = sdist(number_of_neighbouring_samples);  %% Check this
	sdist((number_of_neighbouring_samples+1):number_of_samples) = [];
	u = sdist/delxo;
	weights_vec = (1 - u.^3).^3;
	p = wfit(Nxo,Nyo,weights_vec,fit_degree);
	yhat(i) = polyval(p,points_to_estimate(i));
	niter = 1;
	test = 1;
	ynew = yhat(i);	% get a temp variable for iterations
	while test > convergence_tolerance && niter <= max_number_of_iterations
		%do the robust fitting procedure:
        niter = niter + 1;
		yold = ynew;
		resid = Nyo - polyval(p,Nxo);	% calc residuals	
		s = median(abs(resid));
		u = min(abs(resid/(6*s)),1);	% scale so all are between 0 and 1
		r = (1-u.^2).^2;	
		nw = r.*weights_vec;
		p = wfit(Nxo,Nyo,nw,fit_degree);	% get the fit with new weights
		ynew = polyval(p,points_to_estimate(i));	% what is the value at x
		test = abs(ynew - yold);
	end
	% converged - set the value to ynew
	yhat(i) = ynew;
end

function p = wfit(x,y,w,deg)
% This will perform the weighted least squares
n = length(x);
x = x(:);
y = y(:);
w = w(:);
% get matrices
W = spdiags(w,0,n,n);
A = vander(x);
A(:,1:length(x)-deg-1) = [];
V = A'*W*A;
Y = A'*W*y;
[Q,R] = qr(V,0); 
p = R\(Q'*Y); 
p = p';		% to fit MATLAB convention
