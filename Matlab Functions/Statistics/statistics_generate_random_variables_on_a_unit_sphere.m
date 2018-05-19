function X = statistics_generate_random_variables_on_a_unit_sphere(number_of_samples,number_of_dimensions)
% CSSPHRND Generate random variables on the sphere.
%
%   R = CSSPHRND(N,D) Generates N random variables
%   that are distributed on the d-dimensional unit
%   sphere, where d >= 2. 

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


if number_of_dimensions < 2
   error('ERROR: d > = 2')
   return
end

%generate standard normal random variables:
random_numbers = randn(number_of_dimensions,number_of_samples);

%find the magnitude of each row square each element, add and take the square root:
mag = sqrt(sum(random_numbers.^2));

%make a diagonal matrix of them - inverses:
normalizing_diagonal_matrix = diag(1./mag);

%multiply to scale properly:
X = (random_numbers*normalizing_diagonal_matrix)';