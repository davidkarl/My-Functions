function [fhat,bin] = statistics_get_density_histogram(input_vec,number_of_bins)
% CSHISTDEN     Univariate density histogram.
%
%   [FHAT,BIN] = CSHISTDEN(DATA,N)
%
%   This function will plot the univariate density histogram for the
%   data given in X. N represents the number of bins. The default
%   value for N is 10.
% 
%
%   See also CSHERN1D, CSASH, CSFREQPLOY

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox


if nargin == 1
    number_of_bins = 10;
end

n = length(input_vec);
[nu,bin] = hist(input_vec,number_of_bins);
%get widths of bins:
h = bin(2)-bin(1);
fhat = nu/(n*h);

% %plot as density histogram
% bar(bin,nu/(n*h),1)
