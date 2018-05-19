function [x,y] = statistics_generate_strauss_process_sample_points(...
                        xp,yp,delta_inhibition_distance,number_of_points_to_generate,c_inhibition_parameter)
% CSSTRAUSSPROC Generates a Strauss spatial point process.
%
%   [X,Y] = CSSTRAUSSPROC(XP,YP,DELTA,N,C)
%
%   This function returns X and Y locations for a process
%   that exhibits regularity. This follows the Strauss process
%   model. XP and YP are the vertices of the boundary region.
%   DELTA is the inhibition distance. N is the number of
%   points to generate. C is the inhibition parameter that must
%   be between 0 and 1.
%
%   See also CSPOISSPROC, CSBINPROC, CSCLUSTPROC, CSINHIBPROC

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

if delta_inhibition_distance <= 0
    error('Delta must be positive.')
end

if c_inhibition_parameter<0 || c_inhibition_parameter>1
    error('C must be in the interval [0,1].')
end

X = zeros(number_of_points_to_generate,2);

%Generate the first point":
X(1,:) = statistics_get_homogeneous_2D_poisson_data_points(xp,yp,1);

i = 1;

while i<number_of_points_to_generate
	[sx,sy] = statistics_get_homogeneous_2D_poisson_data_points(xp, yp, 1);
	xt = [sx sy ; X(1:i,:)];
    
	%Find the distance between the events:
	dist = pdist(xt);
    
	%Find the distance between the candidate event and the others that have been generated already:
	ind = find(dist(1:i) <= delta_inhibition_distance);
    
	m = length(ind);
	if m == 0
		% Then ok to keep the point - nothing is close.
		i = i+1;
		X(i,:) = [sx, sy];
	elseif rand(1) <= c_inhibition_parameter^m
		% The ok to keep the point.
		i = i+1;
		X(i,:) = [sx, sy];
	end
end

x = X(:,1);
y = X(:,2);