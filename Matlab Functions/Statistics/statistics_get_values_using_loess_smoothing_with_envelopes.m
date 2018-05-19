function [yhat, yhatlo, xlo, yhatup, xup] = ...
                statistics_get_values_using_loess_smoothing_with_envelopes(...
                    x_input_points,y_input_points,xo,alpha_smoothing_factor,fit_degree,flag_robust_estimation)
% CSLOESSENV    Loess smooth with upper and lower envelopes.
%
%   [YHAT,YLO,XLO,YUP,XUP] = CSLOESSENV(X,Y,XO,ALPHA,DEG,FLAG)
%
%   This finds the loess smooth based on the observed data in X and Y
%   at the domain values given in XO. ALPHA is the smoothing parameter
%   and DEG is the degree of the local polynomial fit. 
%   FLAG indicates whether or not the robust loess is used. 
%   FLAG = 1 indicates a robust loess. FLAG = 0 indicates regular loess.
%
%   See also CSLOESS, CSBOOTBAND


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox


if fit_degree ~= 1 && fit_degree ~= 2
	error('Degree of local fit must be 1 or 2')
end
if alpha_smoothing_factor <= 0 || alpha_smoothing_factor >= 1
	error('Alpha must be between 0 and 1')
end
if length(x_input_points) ~= length(y_input_points)
	error('Input vectors do not have the same length.')
end


% make sure these are sorted properly for plotting purposes
[xs,ind] = sort(x_input_points);
ys = y_input_points(ind);
x_input_points = xs;
y_input_points = ys;

%now do the envelopes.
%find the yhat at the observed data values using loess:
if flag_robust_estimation == 1
	yh = statistics_get_values_using_robust_loess_smoothing(x_input_points,y_input_points,x_input_points,alpha_smoothing_factor,fit_degree);
else
	yh = statistics_get_values_using_loess_smoothing(x_input_points,y_input_points,x_input_points,alpha_smoothing_factor,fit_degree);
end
yhat = yh;

%find the residuals:
resid = y_input_points - yh;
%find the positive residuals and corresponding pairs:
indp = find(resid >= 0);
xp = x_input_points(indp);
yp = yh(indp);
rp = resid(indp);
%find the negative residuals and corresponding pairs:
indn = find(resid < 0 );
xn = x_input_points(indn);
yn = yh(indn);
rn = resid(indn);

%smooth the (x,r) pairs:
if flag_robust_estimation == 1	% then do the robust version
	yup = statistics_get_values_using_robust_loess_smoothing(xp,rp,xp,alpha_smoothing_factor,fit_degree);
	ylo = statistics_get_values_using_robust_loess_smoothing(xn,rn,xn,alpha_smoothing_factor,fit_degree);
else
	yup = statistics_get_values_using_loess_smoothing(xp,rp,xp,alpha_smoothing_factor,fit_degree);
	ylo = statistics_get_values_using_loess_smoothing(xn,rn,xn,alpha_smoothing_factor,fit_degree);
end

%Add the smooths to the yhat's to get the upper and lower envelopes
yhatup = yp + yup;
yhatlo = yn + ylo;
xlo = xn;
xup = xp;

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
