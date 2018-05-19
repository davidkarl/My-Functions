function nstd = PDFB_noise_distribution_estimate(number_of_rows, number_of_columns, laplacian_pyamid_filter, diamond_filter, number_of_directional_filter_levels_at_each_scale_vec)
% PDFB_NEST  Estimate the noise standard deviation in the PDFB domain
%
%   nstd = pdfb_nest(nrows, ncols, pfilt, dfilt, nlevs)
%
% Used for PDFB denoising.  For an additive Gaussian white noise of zero
% mean and sigma standard deviation, the noise standard deviation in the
% PDFB domain (in vector form) is sigma * nstd.

%Number of interations:
number_of_iterations = 100;

%First run to get the size of the PDFB
x = randn(number_of_rows, number_of_columns);
y = PDFB_decomposition(x, laplacian_pyamid_filter, diamond_filter, number_of_directional_filter_levels_at_each_scale_vec);
[c, s] = PDFB_to_vec(y);

nstd = zeros(1, length(c));
nlp = s(1, 3) * s(1, 4);	% number of lowpass coefficients
nstd(nlp+1:end) = nstd(nlp+1:end) + c(nlp+1:end).^2;

%Simulate images which are pure randn noise to get an average distribution
%of such a noise among the different subbands
for k = 2:number_of_iterations
    x = randn(number_of_rows, number_of_columns);
    y = PDFB_decomposition(x, laplacian_pyamid_filter, diamond_filter, number_of_directional_filter_levels_at_each_scale_vec);
    [c, s] = PDFB_to_vec(y);
    
    nstd(nlp+1:end) = nstd(nlp+1:end) + c(nlp+1:end).^2;
end

nstd = sqrt(nstd ./ (number_of_iterations - 1));
