function [X,Y,Z,H] = statistics_get_kernel_density_estimator_2D(input_data_mat,x_points_to_estimate,y_points_to_estimate,h)
% CSKERN2D	Bivariate product kernel density estimate.
%
%	[X,Y,Z,H] = CSKERN2D(DATA,GRIDX,GRIDY,H)
%
%	This returns an estimate using the product kernel method.
%	This does the bivariate case only, since the main purpose
%	of it is to plot using MATLAB functions MESH or SURF.
%	DATA is an n x d matrix of observations.
%	GRIDX and GRIDY are the grid sizes.
%	H is an optional 2-element vector of window widths. The
%	default value for H is the value obtained from the Normal Reference Rule.
%
%   EXAMPLE:
%
%   X = randn(100,2);
%   [x,y,z,h] = cskern2d(X,0.3,0.3);
%   surf(x,y,z)
%
%	See also CSKERNMD

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

% use Scott's rule to get the h widths
[number_of_samples,number_of_dimensions] = size(input_data_mat);
if nargin == 3
	% Get the window widths using Normal Reference Rule.
	s = cov(input_data_mat);
	hx = s(1,1)*number_of_samples^(-1/6);
	hy = s(2,2)*number_of_samples^(-1/6);
else
	hx = h(1);
	hy = h(2);
end

% Get the ranges for x and y
minx = min(input_data_mat(:,1));
maxx = max(input_data_mat(:,1));
miny = min(input_data_mat(:,2));
maxy = max(input_data_mat(:,2));
[X,Y] = meshgrid((minx-2*hx):x_points_to_estimate:(maxx+2*hx),(miny-2*hy):y_points_to_estimate:(maxy+2*hy));
x_mesh_to_estimate = X(:);   %put into col vectors
y_mesh_to_estimate = Y(:);
z = zeros(size(x_mesh_to_estimate));	%these will be max values & color

for i = 1:length(x_mesh_to_estimate)	% note that const done at end
	xloc = x_mesh_to_estimate(i)*ones(number_of_samples,1);
	yloc = y_mesh_to_estimate(i)*ones(number_of_samples,1);
	argx = ((xloc-input_data_mat(:,1))/hx).^2;
	argy = ((yloc-input_data_mat(:,2))/hy).^2;
	z(i) = (sum(exp(-.5*(argx+argy))))/(number_of_samples*hx*hy*2*pi);
end

[mm,nn] = size(X);
Z = reshape(z,mm,nn);

H = [hx, hy];
