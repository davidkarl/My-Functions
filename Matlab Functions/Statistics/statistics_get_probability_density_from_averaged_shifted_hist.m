function [fhat,small_bin_edges] = statistics_get_probability_density_from_averaged_shifted_hist(...
                                                            input_vec,window_width,number_of_shifted_histograms)
% CSASH     Univariate averaged shifted histogram.
%
%   [FHAT,BINEDGE] = CSASH(X,H,M)
%
%   This function constructs the probability density estimate from
%   the averaged shifted histogram method.
%   X is the vector of data. H is the window width. M is the number
%   of shifted histograms.
%
%   EXAMPLE:
%   
%   load snowfall
%   [fhat, binedge] = csash(snowfall,14.6,30);
%
%   See also CSHISTDEN, CSFREQPOLY, CSKERN1D

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox

load snowfall
input_vec = snowfall;
window_width = 14.6;
number_of_shifted_histograms = 30;


delta_shift_size = window_width/number_of_shifted_histograms;
number_of_input_samples = length(input_vec);

%Get the mesh:
t0 = 0;
tf = max(input_vec) + 20;
number_of_small_bins = ceil((tf-t0)/delta_shift_size);
small_bin_edges = t0:delta_shift_size:(t0+delta_shift_size*number_of_small_bins);

%Get the bin counts for the smaller binwidth delta:
vk = histc(input_vec,small_bin_edges);
%Put into a vector with m-1 zero bins on either end:
fhat = [zeros(1,number_of_shifted_histograms-1), vk , zeros(1,number_of_shifted_histograms-1)];

%Get the weight vector:
%Create an inline function for the kernel.
kern = inline('(15/16)*(1-x.^2).^2');
ind = (1-number_of_shifted_histograms):(number_of_shifted_histograms-1);
normalized_indices = ind/number_of_shifted_histograms;

%Get the denominator: 
den = sum(kern(normalized_indices));

%Create the weight vector:
wm = number_of_shifted_histograms * (kern(normalized_indices))/den;

%Get the bin heights over smaller bins.
fhatk = zeros(1,number_of_small_bins);
for k = 1:number_of_small_bins
   ind = k:(2*number_of_shifted_histograms+k-2);
   fhatk(k) = sum(wm.*fhat(ind));
end
fhatk = fhatk/(number_of_input_samples*window_width);
bc = t0+((1:k)-0.5)*delta_shift_size;

%To use the stairs plot, we need to use the bin edges.
fhat = [fhatk fhatk(end)];
stairs(small_bin_edges,[fhatk fhatk(end)])












