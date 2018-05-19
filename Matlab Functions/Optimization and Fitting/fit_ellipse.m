function [ x0, cv, ax, ang ] = fit_ellipse( x, flag )
% fits an ellipse to cover a cloud of dots in 2D defined by x
%
% INPUT: 
% - x : n x 2 matrix of datapoints
% - flag : 0 - use x mean and std, 1 - use x limiting points
%
% OUTPUT:
% - x0 : 1 x 2 vector of the ellipse center
% - ax : 1 x 2 vector of the ellipse principal half axes
% - ang : the angle of the ellipsoid rotation (longest half-axis off the
% x-axis, radians)
% 
% Copyright: Yury Petrov, Oculus VR, 01/2014
%

ax = [];
ang = [];

if nargin < 2
    flag = 0;
end

x0 = mean( x );
cv = cov( x );  % get covariance matrix

if flag == 0 % use x moments
    d = sqrt( ( cv(1,1) - cv(2,2) )^2 / 4 + cv(1,2)^2 ); % discriminant
    t = ( cv(1,1) + cv(2,2) ) / 2;
    ax = 2 * sqrt( [ t + d; t - d ] ); % eigenvalues: [ max min ]
    ang = atan2( d - ( cv(1,1) - cv(2,2) ) / 2, cv(1,2) ); % positions the largest eigenvector along the x-axis
elseif flag == 1 % use x limits
    xt = x * V;     % rotate data into eigevectors RF, the largest eigenvector is along the x-axis
    mi = min( xt );
    ma = max( xt );
    x0 = ( ma + mi ) / 2;
    x0 = x0 * V'; % rotate the ellipse position back to the original orientation
    ax = ( ma - mi ) / 2;
end