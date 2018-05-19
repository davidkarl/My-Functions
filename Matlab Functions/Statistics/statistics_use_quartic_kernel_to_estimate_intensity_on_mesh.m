function [X,Y,Z] = statistics_use_quartic_kernel_to_estimate_intensity_on_mesh(events,bounds_mat,kernel_window_width)
% CSINTKERN Kernel method for estimating intensity of spatial point pattern.
%
%   [X,Y,Z] = CSINTKERN(EVENTS,REGION,H)
%
%   This function implements the kernel method for estimating the
%   intensity for a spatial point pattern. It uses the quartic
%   kernel. Edge effects are ignored. The output X, Y, Z can be
%   used in the MATLAB functions MESH or SURF to plot. Z contains
%   the value of the intensity at points given in X and Y. 
%
%   EVENTS contains the locations for the spatial events. This is
%   an n x 2 matrix. The first column contains the x locations and
%   the second column contains the y locations.
%   REGION is a matrix of x and y locations for the vertices of the
%   study region. The first column contains the x locations and the
%   second column contains the y locations.
%   H is the window width for the kernel.

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

[number_of_points,number_of_dimensions] = size(events);

%Get the ranges for x and y, Get a rectangular bounding region:
minx = min(bounds_mat(:,1));
maxx = max(bounds_mat(:,1));
miny = min(bounds_mat(:,2));
maxy = max(bounds_mat(:,2));

%Get the mesh - Get 100 (50 is actually written) linearly spaced points:
xd = linspace(minx,maxx,50);
yd = linspace(miny,maxy,50);
[X,Y] = meshgrid(xd,yd);
%xc=X(:);   %put into col vectors
%yc=Y(:);
meshgrid_points = [X(:), Y(:)];
number_of_meshgrid_points = length(meshgrid_points(:,1));
xt = [[0,0] ; events];
z = zeros(size(X(:)));	
for i = 1:number_of_meshgrid_points	% note that const done at end
	
    %for each point location, s, find the distances that are less than h:
	xt(1,:) = meshgrid_points(i,:);
	
    %find the distances. 
    %First n points in dist are the distances between the point s and the n event locations:
	dist = pdist(xt); %Matlab function
	ind = find(dist(1:number_of_points) <= kernel_window_width);
	t = (1 - dist(ind).^2/kernel_window_width^2).^2;
	z(i) = sum(t);
end
z = z*3/(pi*kernel_window_width);

[mm,nn] = size(X);
Z = reshape(z,mm,nn);

