function prob = statistics_get_multivariate_normal_density_at_points(points_mat,mu_vec,cov_mat)
% CSEVALNORM Multivariate normal probability density function.
%
%	Y = CSEVALNORM(X,MU,COVM) Returns the value of the multivariate
%	probability density function at the locations given in X.
%
%	INPUTS:		X is an n x d matrix of domain locations.
%				MU is a 1 x d vector
%				COVM is a d x d covariance matrix
%
% 	See also CSNORMP, CSNORMC, CSNORMQ

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


[n,d]=size(points_mat);
prob = zeros(n,1);
a=(2*pi)^(d/2)*sqrt(det(cov_mat));
covi = inv(cov_mat);
for i = 1:n
	xc = points_mat(i,:)-mu_vec;
	arg=xc*covi*xc';
	prob(i)=exp((-.5)*arg);
end
prob=prob/a;
