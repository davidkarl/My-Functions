function [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time,number_of_coefficients_updates_done] = ...
    adaptive_filter_SM_NLMS(desired_signal,input_signal,filter_order,initial_coefficients,gamma_bar_upper_error_bound,gamma_regularization)

%   SM_NLMS.m
%       Implements the Set-membership Normalized LMS algorithm for COMPLEX valued data.
%       (Algorithm 6.1 - book: Adaptive Filtering: Algorithms and Practical
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

%Loop over the different iterations:
for iteration_counter = 1:N_number_of_iterations

    %get current part of the signal:
    current_signal = zero_padded_input_signal(iteration_counter:iteration_counter+number_of_coefficients-1,1);
    current_signal = flip(current_signal);
    
    %get current filter output:
    current_coefficients = next_coefficients;
    current_filter_output = current_coefficients' * current_signal;
     
    %get error = desired_signal - current_filter_output:
    current_filter_error = desired_signal(iteration_counter)-current_filter_output;

    %calculate next coefficients vector using SM rule: 
    if abs(current_filter_error) > gamma_bar_upper_error_bound
       mu_convergence_factor = 1 - gamma_bar_upper_error_bound/abs(current_filter_error);
       next_coefficients = current_coefficients + ...
           mu_convergence_factor/ (gamma_regularization+(current_signal'*current_signal)) ...
           * conj(current_filter_error)*current_signal;
       number_of_coefficients_updates_done = number_of_coefficients_updates_done + 1;
    else
       next_coefficients = current_coefficients;
    end
    
%     regressor                   =   prefixedInput(it+(nCoefficients-1):-1:it,1);
%     outputVector(it,1)          =   (coefficientVector(:,it)')*regressor;
%     errorVector(it,1)           =   desired(it)-outputVector(it,1);
%     if abs(errorVector(it,1)) > S.gamma_bar
%        mu = 1-(S.gamma_bar/abs(errorVector(it,1)));
%        nUpdates = nUpdates+1;
%     else
%        mu = 0;
%     end
%     coefficientVector(:,it+1)   =   coefficientVector(:,it)+(...
%                                     (mu/(S.gamma+regressor'*regressor))*...
%                                     conj(errorVector(it,1))*regressor);
    
    %keep track of parameters:
    filter_output_over_time(iteration_counter) = current_filter_output;
    filter_error_over_time(iteration_counter) = current_filter_error;
    filter_coefficients_over_time(:,iteration_counter+1) = next_coefficients;
end


                                
                                
                                