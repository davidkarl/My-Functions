% Example 4.11
% Computational Statistics Handbook with MATLAB, 2nd Edition
% Wendy L. and Angel R. Martinez

%Generate multivariate normal random variables.
R_correlation_matrix = [1 -.8; -.8 1];  % == normalized covariance matrix 
degrees_of_freedom = 5;
number_of_dimensions = 2;
number_of_samples = 500; 

%Generate n 2-D multivariate normal random variables, centered at 0, with covariance C:
C = chol(R_correlation_matrix);
X_multivariate_normal = randn(number_of_samples,number_of_dimensions)*C;

%Generate chi-square random variables and divide by the degrees of freedom:
X_chi_square = sqrt(chi2rnd(degrees_of_freedom,number_of_samples,1)./degrees_of_freedom);

%Divide to get the multivariate t random variables:
Xt = X_multivariate_normal./repmat(X_chi_square(:),1,number_of_dimensions);

%Do a scatterplot:
plot(Xt(:,1),Xt(:,2),'.');
xlabel('X_1');ylabel('X_2');
title('Multivariate t Random Variables with \nu = 5')

% Check the correlation using the following:
corrcoef(Xt)

 