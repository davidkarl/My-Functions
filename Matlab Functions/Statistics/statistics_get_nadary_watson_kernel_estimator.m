function yhatnw = statistics_get_nadary_watson_kernel_estimator(x_input_points,y_input_points,window_width)
% CSNARDWATS  Nadarya-Watson kernel estimator.
%
%   YHAT = CSNARDWATS(X,Y,H)
%   Performs nonparametric regression using the Nadarya-Watson
%   kernel estimator. The observed data are contained in X and Y.
%   The smoothing parameter is given by the kernel bandwidth H.
%   YHAT contains the values of the smooth for each value in X.
%
%
%   EXAMPLE:
%   
%   load salmon
%   x = salmon(:,1);
%   y = salmon(:,2);
%   [xs,inds] = sort(x);
%   ys = y(inds);
%   yhat = csnardwats(xs,ys,60);
%   plot(xs,ys,'.',xs,yhat)
%
%   See also CSLOESS, CSLOCPOLY, CSLOCLIN, CSLOESSR

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

% Create an inline function to evaluate the weights.
mystrg = '(2*pi*h^2)^(-1/2)*exp(-0.5*((x - mu)/h).^2)';
wfun = inline(mystrg);

%Set up the space to store the estimated values.
%We will get the estimate at all values of x.
yhatnw = zeros(size(x_input_points));
n = length(x_input_points);

%find smooth at each value in x
for i = 1:n
		w = wfun(window_width,x_input_points(i),x_input_points);
		yhatnw(i) = sum(w.*y_input_points)/sum(w);
end






