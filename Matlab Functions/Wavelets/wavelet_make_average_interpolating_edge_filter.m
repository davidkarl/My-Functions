function edge_filter = wavelet_make_average_interpolating_edge_filter(degree_of_polynomial)
% MakeAIBdryFilter -- Edge filters for Average-Interpolating Wavelets
%  Usage
%    EdgeFilt = MakeAIBdryFilter(D)
%  Inputs
%    D         Degree of polynomial for average-interpolation.
%              Must be an even integer.
%  Outputs
%    EdgeFilt  Edge Filter suitable for use by
%              AIRefine, FWT_AI, IWT_AI, AIRefine2d, etc.
%  Description
%    Calculates average-interpolating filters of various orders
%    which may be used with FWT_AI and related tools.
%
%  See Also
%    AIRefine, FWT_AI, AIRefine2d
%

if rem(degree_of_polynomial,2) || degree_of_polynomial < 2,
   edge_filter = [];
else

	%step 1. Moment matrix Mmat (WHAT IS THIS?):
	moment_matrix = zeros(degree_of_polynomial+1,degree_of_polynomial+1);
	for kp1 = 1:(degree_of_polynomial+1),
	   for lp1 = 1:(degree_of_polynomial+1),
			moment_matrix(lp1,kp1) = (lp1^kp1 - (lp1-1)^kp1)/kp1;
	   end
	end
	moment_matrix_inverse = inv(moment_matrix);

	%step 2. Imputation matrx Jmat (WHAT IS THIS?):
	imputation_matrix = zeros(degree_of_polynomial,degree_of_polynomial+1);
	for kpp1 = 1:(degree_of_polynomial+1),
	  for kp1 = 1:(degree_of_polynomial),
			imputation_matrix(kp1,kpp1) = 2 * ((kp1/2)^kpp1 - (kp1/2 - .5)^kpp1)/(kpp1);
	  end
	end

	%step 3. Compose for prediction matrix (WHAT IS THIS?):
	prediction_matrix = imputation_matrix * moment_matrix_inverse ;
	edge_filter = prediction_matrix;
end


%
% For Article "Smooth Wavelet Decompositions with Blocky Coefficient Kernels"
% Copyright (c) 1993. David L. Donoho
%
