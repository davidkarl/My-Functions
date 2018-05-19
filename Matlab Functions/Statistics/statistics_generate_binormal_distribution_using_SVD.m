% Example 5.12
% Computational Statistics Handbook with MATLAB, 2nd Edition
% Wendy L. and Angel R. Martinez

%Create a positive definite covariance matrix:
covariance_mat = [2, 1.5; 1.5, 9];

%Create mean at (2,3):
mu = [2 3];

%do SVD to covariance mat:
[u,s,v] = svd(covariance_mat);
variance_sqrt_mat = (v*(u'.*sqrt(s)))';

%Get standard normal random variables: 
normal_random_vec = randn(250,2); 

%Use x=z*sigma+mu to transform - see Chapter 4:
data = normal_random_vec*variance_sqrt_mat + ones(250,1)*mu; 

%Create a scatter plot using the plot function:
plot(data(:,1),data(:,2),'x')
axis equal

%Create a scatter plot using the scatter. 
%Use filled-in markers.
scatter(data(:,1),data(:,2),'filled')
axis equal
box on
