function Filter = wavelet_make_average_interpolating_filter(degree_of_polynomial)
% MakeAIFilter --  Filters for Average-Interpolating Wavelets
%  Usage
%    Filt = MakeAIFilter(D)
%  Inputs
%    D         Degree of polynomial for average-interpolation.
%              Must be an even integer.
%  Outputs
%    Filt      Average-Interpolating Filter suitable for use by
%              AIRefine, FWT_AI, IWT_AI, AIRefine2d, etc.
%
%  Description
%    Calculates average-interpolating filters of various orders
%    which may be used with FWT_AI and related tools.
%
%  See Also
%    AIRefine, FWT_AI, AIRefine2d
%

if rem(degree_of_polynomial,2) || degree_of_polynomial < 2,
   Filter = [];
else

	% step 1. Moment matrix Mmat:
	moment_matrix = zeros(degree_of_polynomial+1,degree_of_polynomial+1);
	for kp1 = 1:(degree_of_polynomial+1),
	   for lp1 = (-degree_of_polynomial/2+1):(degree_of_polynomial/2+1),
			moment_matrix(lp1+degree_of_polynomial/2,kp1) = (lp1^kp1 - (lp1-1)^kp1)/kp1;
	   end
	end
	moment_matrix_inverse = inv(moment_matrix);

	% step 2. Imputation matrix Jmat:
	imputation_matrix = zeros(2,degree_of_polynomial+1);
	for kpp1 = 1:(degree_of_polynomial+1),
	  for kp1 = 1:2,
			imputation_matrix(kp1,kpp1) = 2 * ((kp1/2)^kpp1 - (kp1/2 - .5)^kpp1)/(kpp1);
	  end
	end

	% step 3. Compose for prediction matrix:
	prediction_matrix = imputation_matrix * moment_matrix_inverse; 
    prediction_matrix = prediction_matrix(2:-1:1,:);
	Filter = prediction_matrix(:);

end
    
% For Article "Smooth Wavelet Decompositions with Blocky Coefficient Kernels"
% Copyright (c) 1993. David L. Donoho
