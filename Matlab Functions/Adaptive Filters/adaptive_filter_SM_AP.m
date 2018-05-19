function    [filter_final_output_vec,filter_final_error_vec,filter_coefficients_over_time,number_of_coefficients_updates_done] = ...
    adaptive_filter_SM_AP(desired_signal,input_signal,filter_order,initial_coefficients,gamma_bar_error_bound,gamma_bar_error_bound_vec,gamma_regularization,L_samples_in_memory)

%   SM_AP.m
%       Implements the Set-membership Affine-Projection algorithm for COMPLEX valued data.
%       (Algorithm 6.2 - book: Adaptive Filtering: Algorithms and Practical
%                                                       Implementation, Diniz)

%initialize parameters:
number_of_coefficients = filter_order+1;
N_number_of_iterations = length(desired_signal);
number_of_coefficients_updates_done = 0;

%check that gamma_bar_error_bound_vec's length is the same as the number of membership cards + 1:
if length(gamma_bar_error_bound_vec) ~= (L_samples_in_memory+1)
   fprintf('S.gamma_barVector must have (S.memoryLength + 1) rows. \n \n');
   return
end

%check that each error bound in the error bound vec is smaller then currently to use error bound:
for k=1:(L_samples_in_memory + 1)
   if abs(gamma_bar_error_bound_vec(k)) > gamma_bar_error_bound
      fprintf('S.gamma_barVector(%i) must be chosen less or equal to S.gamma_bar (in modulus). \n \n',k);
   return
   end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initialize error, output and coefficients vectors:
filter_error_over_time_conj = zeros(L_samples_in_memory+1, N_number_of_iterations);
filter_output_over_time_conj = zeros(L_samples_in_memory+1, N_number_of_iterations);
filter_coefficients_over_time = zeros(number_of_coefficients ,N_number_of_iterations+1);
filter_coefficients_over_time(:,1) = initial_coefficients;

current_signal_matrix = zeros(number_of_coefficients, L_samples_in_memory+1);

%initialize auxiliary next coefficients vector:
next_coefficients = initial_coefficients;

%Improve source code regularity
zero_padded_input_signal_matrix = [zeros(number_of_coefficients-1,1) ; input_signal];
zero_padded_desired_signal_matrix = [zeros(L_samples_in_memory,1) ; desired_signal];
I = eye(L_samples_in_memory+1);

%Loop over the different iterations:
for iteration_counter = 1:N_number_of_iterations

    %get current part of the signal:
    current_signal_matrix = circshift(current_signal_matrix,[0,1]);
    current_signal = zero_padded_input_signal_matrix(iteration_counter:iteration_counter+number_of_coefficients-1,1);
    current_signal_matrix(:,1) = flip(current_signal);
    
    %get current filter output:
    current_coefficients = next_coefficients;
    current_filter_output_vec_conj = current_signal_matrix' * current_coefficients;
    
    %get error = desired_signal - current_filter_output:
    current_desired_signal_vec = zero_padded_desired_signal_matrix(iteration_counter:iteration_counter+L_samples_in_memory);
    current_filter_error_vec_conj = conj(flip(current_desired_signal_vec)) - current_filter_output_vec_conj;

    %calculate next coefficients vector using affine projection rule:
    if abs(current_filter_error_vec_conj) > gamma_bar_error_bound
        next_coefficients = current_coefficients + (current_signal_matrix * inv(current_signal_matrix'*current_signal_matrix + gamma_regularization*I) * (current_filter_error_vec_conj - conj(gamma_bar_error_bound_vec)));
    else
        next_coefficients = current_coefficients;
    end
    
    %keep track of parameters:
    filter_output_over_time_conj(:,iteration_counter) = current_filter_output_vec_conj;
    filter_error_over_time_conj(:,iteration_counter) = current_filter_error_vec_conj;
    filter_coefficients_over_time(:,iteration_counter+1) = next_coefficients;
end
filter_final_output_vec = filter_output_over_time_conj(1,:)';
filter_final_error_vec = filter_error_over_time_conj(1,:)';





