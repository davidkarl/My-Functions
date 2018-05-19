function [smoothed_response_vec_axis_points,smoothed_response_vec] = statistics_smooth_bins(predictor_vec,response_vec,number_of_bins)

% CSBINSMTH Bin smoother
%
%	[CK,YHAT] = CSBINSMTH(X,Y,K) constructs a bin smoother for the predictor
%	values X and the response values Y. The smoothness of the resulting
%	nonparametric regression is governed by the number of regions or bins
%	K.
%
%	INPUTS:		X is a vector of predictor values
%				Y is a vector of response values
%				K is the number of bins
%
%   The ouput of this function is a vector YHAT of smoothed values over the
%   bins and the bins CK.
%
%   A plot of the data points and smooth is also produced by this function.
%
%   EXAMPLE:
%
%   load vineyard
%   y = totlugcount;
%   x = row;
%   [ck,yhat] = csbinsmth(x,y,10);
%
%   See also CSLOESS, CSLOCPOLY, CSLOCLIN, CSNARDWATS, CSLOESSR,
%   CSRMEANSMTH, CSSPLINESMTH, CSBINSMTH

%   W. L. and A. R. Martinez, 12/14/07
%   Computational Statistics Toolbox, V 2 

% To smooth using the bin smoother, one sets the number of bins first,
% and then chooses the cutoff points so each region has approximately the
% same number of data points. So, the number of bins is the smoothing
% parameter, but this gives rise to the span - the proportion of points in
% each neighborhood. 

load vineyard;
predictor_vec = totlugcount;
response_vec = row;
number_of_bins = 10;

%Set the number of bins and the number of observations in each one:
number_of_samples = length(predictor_vec);
number_of_samples_per_bin_vec = ones(1,number_of_bins) * floor(number_of_samples/number_of_bins);
rm = rem(number_of_samples,number_of_bins);
if rm ~= 0
    number_of_samples_per_bin_vec(1:rm) = number_of_samples_per_bin_vec(1:rm) + 1;
end
number_of_samples_per_bin_vec_cumsumed = cumsum(number_of_samples_per_bin_vec);

%Sort the data:
[predictor_vec_sorted,inds] = sort(predictor_vec);
response_vec_sorted = response_vec(inds);

%Find the c_k and the value of the smooth in each:
smoothed_response_vec_axis_points = predictor_vec_sorted(1);
smoothed_response_vec_axis_points(number_of_bins+1) = predictor_vec_sorted(end);
smoothed_response_vec = zeros(1,number_of_bins+1);
smoothed_response_vec(1) = mean(response_vec_sorted(1:number_of_samples_per_bin_vec_cumsumed(1)));
for i = 1:(number_of_bins-1)
    smoothed_response_vec_axis_points(i+1) = mean(predictor_vec_sorted(number_of_samples_per_bin_vec_cumsumed(i):number_of_samples_per_bin_vec_cumsumed(i)+1));
    smoothed_response_vec(i+1) = mean(response_vec_sorted(number_of_samples_per_bin_vec_cumsumed(i)+1:number_of_samples_per_bin_vec_cumsumed(i+1)));
end
smoothed_response_vec(end) = smoothed_response_vec(end-1);

plot(predictor_vec,response_vec,'o');
xlabel('X'),ylabel('Y')
hold on
stairs(smoothed_response_vec_axis_points,smoothed_response_vec)
hold off

