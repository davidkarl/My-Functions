function [X,Y] = statistics_generate_2D_poisson_process(lambda_parameter, region_string_circle_or_rect, region_parameters)
% CSPOISSPROC Generate a 2-D Poisson process.
%
%   [X,Y] = CSPOISSPROC(LAMBDA, REGION, DIMS)
%
%   This function will return the points X and Y that
%   represent locations for a 2-D Poisson process.
%
%   The user has a choice between one generated over a
%   circular region centered at 0 or one generated over a
%   rectangular region. If a circular region is used, then the
%   radius must be given. If a rectangular region is used, then
%   the dimensions of the rectangle must be given.
%
%   LAMBDA is the rate.
%   REGION is a string specifying the type of region. The choices
%   are 'circle' or 'rect'.
%   DIMS is either the radius of the circle or the width and height
%   of the rectangle.
%
%   See also CSBINPROC, CSCLUSTPROC, CSINHIBPROC, CSSTRAUSPROC

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

if lambda_parameter <= 0 
    error('Lambda must be greater than 0')
end

if ~(strcmp(region_string_circle_or_rect,'circle') | strcmp(region_string_circle_or_rect,'rect'))
    error('The choices for the region are ''circle'' or ''rect''.')
end

switch region_string_circle_or_rect 
case 'circle'
    if length(region_parameters) ~= 1
        error('DIMS must be the scalar radius for circular regions.')
    end
    r = region_parameters;
	% Circle one.
	tol = 0;
	i=1;
	while tol < pi*r^2
		x(i) = exprnd(1/lambda_parameter,1,1);
		tol = sum(x);
		i=i+1;
	end
	x(end)=[];
	N = length(x);
	th = 2*pi*rand(1,N);
	R = zeros(1,N);
	for i = 1:N
		R(i) = sqrt(sum(x(1:i))/pi);
	end
	[X,Y]=pol2cart(th,R);
otherwise
    if length(region_parameters) ~= 2
        error('DIMS must have the width and height of the rectangular region.')
    end
    % Rectangle
	a = region_parameters(1)*region_parameters(2);
	c = region_parameters(2);
	tol = 0;
	i = 1;
	while tol < a
		x(i) = exprnd(1/lambda_parameter,1,1);
		tol = sum(x);
		i = i+1;
	end
	x(end) = [];
	N = length(x);
	X = zeros(1,N);
	Y = rand(1,N)*c;
	for i = 1:N
		X(i) = sum(x(1:i))/c;
	end
end
