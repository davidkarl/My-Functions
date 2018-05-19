function X_data_mat_with_structure_removed = statistics_projection_pursuit_structure_removal(Z_input_data_mat,a,b)
% CSPPSTRTREM Projection pursuit structure removal.
%
%   X = CSPPSTRTREM(Z,ALPHA,BETA) Removes the structure
%   in a projection to the plane spanned by ALPHA and BETA.
%   Usually, this plane is found using projection pursuit EDA.
%   
%   The input matrix Z is an n x d matrix of spherized observations,
%   one for each row. The output matrix X is the data with the
%   structure removed.
%
%   See also CSPPEDA, CSPPIND

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


%just do this 5 times:
max_number_of_iterations = 5;	% maximum number of iterations allowed
[number_of_samples,number_of_dimensions] = size(Z_input_data_mat);

%find the orthonormal matrix needed via Gram-Schmidt:
U = eye(number_of_dimensions,number_of_dimensions);   
%fill in top two rows with plane vectors:
U(1,:) = a';	% vector for best plane
U(2,:) = b';
%do Grahm-Schmidt process:
for i = 3:number_of_dimensions
   for j = 1:(i-1)
      U(i,:) = U(i,:) - (U(j,:)*U(i,:)') * U(j,:);
   end
   U(i,:) = U(i,:)/sqrt(sum(U(i,:).^2));
end

%Transform data using the matrix U:
T = U*Z_input_data_mat';	% to match Friedman's treatment. T is d x n
x_2D_projection1 = T(1,:);	% These should be the 2-d projection that is 'best'
x_2D_projection2 = T(2,:);

%Gaussianize the first two rows of T 
%set of vector of angles:
gamma_angles = [0, pi/4, pi/8, 3*pi/8];
for iteration_counter = 1:max_number_of_iterations
   %gaussianize the data:
   for i = 1:length(gamma_angles)
       
      %rotate about origin:
      x_2D_projection_rotated1 = x_2D_projection1*cos(gamma_angles(i)) + x_2D_projection2*sin(gamma_angles(i));
      x_2D_projection_rotated2 = x_2D_projection2*cos(gamma_angles(i)) - x_2D_projection1*sin(gamma_angles(i));
      
      %Transform to normality:
      [iteration_counter, sorted_indices_list1] = sort(x_2D_projection_rotated1);  % get the ranks
      [iteration_counter, sorted_indices_list2] = sort(x_2D_projection_rotated2);
      
      %Get quantiles of each term of the 2D projected vec:
      arg1 = (sorted_indices_list1-0.5)/number_of_samples;	% get the arguments
      arg2 = (sorted_indices_list2-0.5)/number_of_samples;
      
      %switch between current 2D projection value and value(quantile(term)) to get normality:
      x_2D_projection1 = norminv(arg1,0,1); % transform to normality
      x_2D_projection2 = norminv(arg2,0,1);
   end
end

%Set the first two rows of T to the Gaussianized values:
T(1,:) = x_2D_projection1;
T(2,:) = x_2D_projection2;

%Data mat with structure removed:
X_data_mat_with_structure_removed = (U'*T)';




