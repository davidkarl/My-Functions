function    [filter_final_output_vec,filter_final_error_vec,filter_coefficients_over_time] = ...
    adaptive_filter_affine_projection(desired_signal,input_signal,filter_order,initial_coefficients,gamma_normalizing_factor,L_samples_in_memory,mu_convergence_factor)

%   Affine_projection.m
%       Implements the Complex Affine-Projection algorithm for COMPLEX valued data.

%initialize parameters:
number_of_coefficients = filter_order+1;
N_number_of_iterations = length(desired_signal);

%initialize error, output and coefficients vectors:
filter_error_over_time = zeros(L_samples_in_memory+1, N_number_of_iterations);
filter_output_over_time = zeros(L_samples_in_memory+1, N_number_of_iterations);
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
for iteration_counter = 1:N_number_of_iterations,

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
    next_coefficients = current_coefficients + ...
                                    (mu_convergence_factor * current_signal_matrix * ...
                                    inv((current_signal_matrix'*current_signal_matrix)+gamma_normalizing_factor*I) * ...
                                    current_filter_error_vec_conj);
                                
    %keep track of parameters:
    filter_output_over_time(:,iteration_counter) = current_filter_output_vec_conj;
    filter_error_over_time(:,iteration_counter) = current_filter_error_vec_conj;
    filter_coefficients_over_time(:,iteration_counter+1) = next_coefficients;
end
filter_final_output_vec = filter_output_over_time(1,:)';
filter_final_error_vec = filter_error_over_time(1,:)';




