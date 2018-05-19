function    [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time] = ...
    adaptive_filter_power_of_2(desired_signal,input_signal,filter_order,initial_coefficients,data_number_of_bits,tau_gain_factor,mu_convergence_factor)

%   Power2_error.m
%       Implements the Power-of-Two Error LMS algorithm for REAL valued data.

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
    if (abs(current_filter_error)>=1)
        current_filter_error_power2 = sign(current_filter_error); 
    elseif abs(current_filter_error)<2^(-data_number_of_bits+1)
        current_filter_error_power2 = tau_gain_factor * sign(current_filter_error);
    else
        current_filter_error_power2 = 2^(floor(log2(abs(current_filter_error)))) * sign(current_filter_error);
    end
    
    
    %calculate next coefficients vector using power of 2 rule:
    next_coefficients = current_coefficients + ...
                                    (2 * mu_convergence_factor * conj(current_filter_error_power2) * current_signal);
                                
    %keep track of parameters:
    filter_output_over_time(iteration_counter) = current_filter_output;
    filter_error_over_time(iteration_counter) = current_filter_error;
    filter_coefficients_over_time(:,iteration_counter+1) = next_coefficients;
end

