function [as,bs,projection_pursuit_max_overall] = statistics_projection_pursuit_exploratory_data_analysis(...
                                Z_input_data_mat, ...
                                size_of_search_neighborhood, ...
                                half_number_of_steps_to_halve_neighborhood, ...
                                number_of_random_starts)
% CSPPEDA Projection pursuit exploratory data analysis.
%
%   [ALPHA,BETA,PPM] = CSPPEDA(Z,C,HALF,M)
%
%   This function implements projection pursuit exploratory
%   data analysis using the chi-square index. 
%
%   Z is an n x d matrix of observations that have been sphered!!!!!.
%   C is the size of the starting neighborhood for each search.
%   HALF is the number of steps without an increase in the index,
%   at which time the neighborhood is halved.
%   M is the number of random starts.
%
%   This uses the method of Posse. See the M-file for the references.
%
%   See also CSPPIND, CSPPSTRTREM

%   References: 
%   Christian Posse. 1995. 'Projection pursuit explortory
%   data analysis,' Computational Statistics and Data Analysis, vol. 29,
%   pp. 669-687.
%   Christian Posse. 1995. 'Tools for two-dimensional exploratory
%   projection pursuit,' J. of Computational and Graphical Statistics, vol 4
%   pp. 83-100

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 


% get the necessary constants
[number_of_samples,number_of_dimensions] = size(Z_input_data_mat);
max_number_of_iterations = 1500;
search_window_size = size_of_search_neighborhood;
search_tolerance = 0.00001;
search_tolerance = 0.01;
as = zeros(number_of_dimensions,1);	% storage for the information
bs = zeros(number_of_dimensions,1);
projection_pursuit_max_overall = realmin;


%find the probability of bivariate standard normal over each radial box:
fnr = inline('r.*exp(-0.5*r.^2)','r');
ck = ones(1,40);
ck(1:8) =   1/8 * quadl(fnr, 0,                  sqrt(2*log(6))/5);
ck(9:16) =  1/8 * quadl(fnr, sqrt(2*log(6))/5,   2*sqrt(2*log(6))/5);
ck(17:24) = 1/8 * quadl(fnr, 2*sqrt(2*log(6))/5, 3*sqrt(2*log(6))/5);
ck(25:32) = 1/8 * quadl(fnr, 3*sqrt(2*log(6))/5, 4*sqrt(2*log(6))/5);
ck(33:40) = 1/8 * quadl(fnr, 4*sqrt(2*log(6))/5, 5*sqrt(2*log(6))/5);



for random_start_iteration_counter = 1:number_of_random_starts  % m 
   %Generate a random starting plane, this will be the current best plane:
   a_random_multi_dimensional_point = randn(number_of_dimensions,1);
   vec_magnitude = sqrt(sum(a_random_multi_dimensional_point.^2));
   a_normalized = a_random_multi_dimensional_point/vec_magnitude;
   b_random_multi_dimensional_point = randn(number_of_dimensions,1);
   b_orthogonal_to_a = b_random_multi_dimensional_point - (a_normalized'*b_random_multi_dimensional_point)*a_normalized;
   vec_magnitude = sqrt(sum(b_orthogonal_to_a.^2));
   b_normalized = b_orthogonal_to_a/vec_magnitude;
      
   %find the projection index for this plane, this will be the initial value of the index:
   pp_index_max_current_random_start = statistics_chi_square_projection_pursuit_index(...
                    Z_input_data_mat,a_normalized,b_normalized,number_of_samples,ck);
   
   %keep repeating this search until the value c becomes less than cstop 
   %or until the number of iterations exceeds maxiter:
   iteration_counter = 0;		% number of iterations
   number_of_iterations_without_increase_in_index = 0;	% number of iterations without increase in index
   size_of_search_neighborhood = search_window_size;
   while (iteration_counter < max_number_of_iterations) && (size_of_search_neighborhood > search_tolerance)	
      disp(['Iter = ' int2str(iteration_counter) ...
      '   ...c = ' num2str(size_of_search_neighborhood) ...
      '      Index = ' num2str(pp_index_max_current_random_start) ...
      '      i = ' int2str(random_start_iteration_counter)])
      
      %generate a p-vector on the unit sphere to transfer plane to check it:
      v = randn(number_of_dimensions,1);
      vec_magnitude = sqrt(sum(v.^2));
      v_sphered = v/vec_magnitude;
      
      %find the a1,b1 and a2,b2 planes:
      t = a_normalized + size_of_search_neighborhood*v_sphered;
      vec_magnitude = sqrt(sum(t.^2));
      a1 = t/vec_magnitude;
      t = a_normalized - size_of_search_neighborhood*v_sphered;
      vec_magnitude = sqrt(sum(t.^2));
      a2 = t/vec_magnitude;
      
      t = b_normalized - (a1'*b_normalized)*a1;
      vec_magnitude = sqrt(sum(t.^2));
      b1 = t/vec_magnitude;
      t = b_normalized - (a2'*b_normalized)*a2;
      vec_magnitude = sqrt(sum(t.^2));
      b2 = t/vec_magnitude;
      
      %check projection pursuit index for two new planes:
      ppi1 = statistics_chi_square_projection_pursuit_index(Z_input_data_mat,a1,b1,number_of_samples,ck);
      ppi2 = statistics_chi_square_projection_pursuit_index(Z_input_data_mat,a2,b2,number_of_samples,ck);
      [max_projection_pursuit_index,max_projection_pursuit_index_1_or_2] = max([ppi1,ppi2]);
      
      if max_projection_pursuit_index > pp_index_max_current_random_start	
          %then reset plane and index to this value:
          eval(['a_normalized=a' int2str(max_projection_pursuit_index_1_or_2) ';']);
          eval(['b_normalized=b' int2str(max_projection_pursuit_index_1_or_2) ';']);
          eval(['pp_index_max_current_random_start=ppi' int2str(max_projection_pursuit_index_1_or_2) ';']);
      else
          %no increase:
          number_of_iterations_without_increase_in_index = number_of_iterations_without_increase_in_index+1;	% no increase
      end
      
      if number_of_iterations_without_increase_in_index == half_number_of_steps_to_halve_neighborhood	
          %then decrease the neighborhood:
          size_of_search_neighborhood = size_of_search_neighborhood * 0.5;
          number_of_iterations_without_increase_in_index=0;
      end
      
      %uptick iteration counter:
      iteration_counter = iteration_counter+1;
   end %end iteration counter loop
   
   if pp_index_max_current_random_start > projection_pursuit_max_overall
       % save the current projection as a best plane
       as = a_normalized;
       bs = b_normalized;
       projection_pursuit_max_overall = pp_index_max_current_random_start;
   end
end


   
   