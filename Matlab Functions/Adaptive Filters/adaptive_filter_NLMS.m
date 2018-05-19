function    [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time] = ...
    adaptive_filter_NLMS(desired_signal,input_signal,filter_order,initial_coefficients,gamma_normalization_factor,mu_convergence_factor)

%   NLMS.m
%       Implements the Normalized LMS algorithm for COMPLEX valued data.


%initialize parameters:
number_of_coefficients = filter_order+1;
N_number_of_iterations = length(desired_signal);

%initialize error, output and coefficients vectors:
filter_error_over_time = zeros(N_number_of_iterations,1);
filter_output_over_time = zeros(N_number_of_iterations,1);
filter_coefficients_over_time = zeros(number_of_coefficients ,N_number_of_iterations+1);
filter_coefficients_over_time(:,1) = initial_coefficients;

%initialize auxiliary next coefficients vector:
next_coefficients = initial_coefficients;

%Improve source code regularity
zero_padded_input_signal = [zeros(number_of_coefficients-1,1) ; input_signal];

%Loop over the different iterations:
for iteration_counter = 1:N_number_of_iterations,

    %get current part of the signal:
    current_signal = zero_padded_input_signal(iteration_counter:iteration_counter+number_of_coefficients-1,1);
    current_signal = flip(current_signal);
    
    %get current filter output:
    current_coefficients = next_coefficients;
    current_filter_output = current_coefficients' * current_signal;
    
    %get error = desired_signal - current_filter_output:
    current_filter_error = desired_signal(iteration_counter)-current_filter_output;

    %calculate next coefficients vector using NLMS rule:
    next_coefficients = current_coefficients + ...
           (mu_convergence_factor * conj(current_filter_error) * current_signal) / (gamma_normalization_factor + current_signal'*current_signal);
                                
    %keep track of parameters:
    filter_output_over_time(iteration_counter) = current_filter_output;
    filter_error_over_time(iteration_counter) = current_filter_error;
    filter_coefficients_over_time(:,iteration_counter+1) = next_coefficients;
end

