function [x,y] = statistics_get_homogeneous_2D_poisson_data_points(x_vertices, y_vertices, number_of_data_points)
% CSBINPROC Generate homogeneous 2-D Poisson process.
%
%   [X,Y] = CSBINPROC(XP,YP,N) This function generates a
%   homogeneous 2-D Poisson process. Conditional on the number
%   of data points N, this is uniformly distributed over the
%   study region. The vectors XP and YP correspond to the x and y
%   vertices of the study region. The vectors X and Y contain
%   the locations for the generated events.
%
%   EXAMPLE:
%   
%   xp = [0 1 1 0];  % vertices for region
%   yp = [0 0 1 1];
%   [X,Y] = csbinproc(xp,yp,100);
%   plot(X,Y,'.')
%
%   See also CSPOISSPROC, CSCLUSTPROC, CSINHIBPROC, CSSTRAUSPROC

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox


x = zeros(number_of_data_points,1);
y = zeros(number_of_data_points,1);
i = 1;

%find the maximum and the minimum for a 'box' around the region. 
%Will generate uniform on this, and throw out those points that are not inside the region.
min_x = min(x_vertices);
max_x = max(x_vertices);
min_y = min(y_vertices);
max_y = max(y_vertices);
area_size_x = max_x - min_x;
area_size_y = max_y - min_y;

while i <= number_of_data_points
	xt = rand(1)*area_size_x + min_x;
	yt = rand(1)*area_size_y + min_y;
	k = inpolygon(xt, yt, x_vertices, y_vertices);
	if k == 1
		%if it's in the region:
		x(i) = xt;
		y(i) = yt;
		i = i+1;
	end
end






	