function [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time,number_of_coefficients_updates_done] = ...
    adaptive_filter_SM_BNLMS(desired_signal,input_signal,filter_order,initial_coefficients,gamma_bar_upper_error_bound,gamma_normalization_factor)

%   SM_BNLMS.m
%       Implements the Set-membership Binormalized LMS algorithm for REAL valued data.
%       (Algorithm 6.5 - book: Adaptive Filtering: Algorithms and Practical
%                                                       Implementation, Diniz)

%initialize parameters:
number_of_coefficients = filter_order+1;
N_number_of_iterations = length(desired_signal);
number_of_coefficients_updates_done = 0;

%initialize error, output and coefficients vectors:
filter_error_over_time = zeros(N_number_of_iterations,1);
filter_output_over_time = zeros(N_number_of_iterations,1);
filter_coefficients_over_time = zeros(number_of_coefficients ,N_number_of_iterations+1);
filter_coefficients_over_time(:,1) = initial_coefficients;

%initialize auxiliary next coefficients vector:
next_coefficients = initial_coefficients;

%Improve source code regularity
zero_padded_input_signal = [zeros(number_of_coefficients-1,1) ; input_signal];

%initialize current and previous signals:
first_signal = zeros(number_of_coefficients,1);
second_signal = zeros(number_of_coefficients,1);


%Loop over the different iterations:
for iteration_counter = 1:N_number_of_iterations

    %uptick two signal window:
    first_signal = second_signal;
    second_signal = zero_padded_input_signal(iteration_counter:iteration_counter+number_of_coefficients-1,1);
    second_signal = flip(second_signal);
    
    %get current filter output:
    current_coefficients = next_coefficients;
    current_filter_output = current_coefficients.' * second_signal;
    
    %get error = desired_signal - current_filter_output:
    current_filter_error = desired_signal(iteration_counter)-current_filter_output;
    
    %calculate next coefficients vector using SM rule:
    if abs(current_filter_error) > gamma_bar_upper_error_bound
       mu_convergence_factor = 1 - gamma_bar_upper_error_bound/abs(current_filter_error);
       number_of_coefficients_updates_done = number_of_coefficients_updates_done + 1;
    else
       mu_convergence_factor = 0;
    end
    
    %get lambda1 & lambda2:
    lambda1 = (mu_convergence_factor * current_filter_error * (norm(first_signal,2))^2) /...
              ( gamma_normalization_factor + norm(second_signal,2)^2 * norm(first_signal,2)^2 - (first_signal.'*second_signal)^2 );
    lambda2 =-(mu_convergence_factor * current_filter_error * (first_signal.'*second_signal) )/...
              ( gamma_normalization_factor + norm(second_signal,2)^2 * norm(first_signal,2)^2 - (first_signal.'*second_signal)^2 );

    %update coefficients:
    next_coefficients = current_coefficients + lambda1*second_signal + lambda2*first_signal;
    
    %keep track of parameters:
    filter_output_over_time(iteration_counter) = current_filter_output;
    filter_error_over_time(iteration_counter) = current_filter_error;
    filter_coefficients_over_time(:,iteration_counter+1) = next_coefficients;

end


