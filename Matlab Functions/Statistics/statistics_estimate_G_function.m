function ghat = statistics_estimate_G_function(input_points_mat,domains_vec)
% CSGHAT Estimate of the G function.
%
%   GHAT = CSGHAT(X,W)
%
%   This returns an estimate of the G function withouth taking into
%   account any edge effects. The input argument X contains the event
%   locations. The input argument W is a vector representing the domain
%   over which to evaluate the function Ghat.
%
%   See also CSFHAT, CSKHAT

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

number_of_samples = length(input_points_mat(:,1));
number_of_points_to_estimate = length(domains_vec); %UNDERSTAND domains_vec better
ghat = zeros(1,number_of_points_to_estimate);

%The G function is the nearest neighbor distances for each event.
%Find the distances for all points.
dist = pdist(input_points_mat); %Matlab function

%convert to a matrix and put large numbers on the diagonal:
D = diag(realmax*ones(1,number_of_samples)) + squareform(dist);

%Find the smallest distances in each row or col.
mind = min(D);

%Now get the values for ghat:
for i = 1:number_of_points_to_estimate
	ind = find(mind<=domains_vec(i));
	ghat(i) = length(ind);
end
ghat = ghat/number_of_samples;