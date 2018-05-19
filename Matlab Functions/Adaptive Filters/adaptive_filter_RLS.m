function    [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time,aposteriori_output_over_time,aposteriori_error_over_time] =  ...
    adaptive_filter_RLS(desired_signal,input_signal,filter_order,delta_initial_autocorrelation_matrix_factor,lambda_smoothing_factor)

%   RLS.m
%       Implements the RLS algorithm for COMPLEX valued data.

%initialize parameters:
number_of_coefficients = filter_order+1;
N_number_of_iterations = length(desired_signal);

%initialize error, output and coefficients vectors:
filter_error_over_time = zeros(N_number_of_iterations,1);
filter_output_over_time = zeros(N_number_of_iterations,1);
filter_coefficients_over_time = zeros(number_of_coefficients , N_number_of_iterations+1);
aposteriori_output_over_time = zeros(N_number_of_iterations,1);
aposteriori_error_over_time = zeros(N_number_of_iterations,1);

%initialize Sd & pd (autocorrelation and crosscorrelation):
Sd_inverse_auto_correlation = delta_initial_autocorrelation_matrix_factor * eye(number_of_coefficients);
pd_signal_noise_cross_correlation = zeros(number_of_coefficients,1);

%Improve source code regularity:
zero_padded_input_signal = [zeros(number_of_coefficients-1,1) ; input_signal];

%Initiailize coefficients:
next_coefficients = Sd_inverse_auto_correlation * pd_signal_noise_cross_correlation;


%Loop over the different iterations:
for iteration_counter = 1:N_number_of_iterations,
    
    %get current part of the signal:
    current_signal = zero_padded_input_signal(iteration_counter:iteration_counter+number_of_coefficients-1,1);
    current_signal = flip(current_signal);
    
    %get current filter output:
    current_coefficients = next_coefficients;
    current_filter_output  =   current_coefficients'*current_signal;

    %get error = desired_signal - current_filter_output:
    current_filter_error   =   desired_signal(iteration_counter)-current_filter_output;

    %update Sd (inverse autocorrelation matrix) using inversion lemma:
    Sd_inverse_auto_correlation = 1/lambda_smoothing_factor *(Sd_inverse_auto_correlation - (Sd_inverse_auto_correlation*(current_signal*current_signal')*Sd_inverse_auto_correlation)...
                    / (lambda_smoothing_factor + current_signal'*Sd_inverse_auto_correlation*current_signal));
    
    %update pd signal-noise crosscorrelation:
    pd_signal_noise_cross_correlation = lambda_smoothing_factor*pd_signal_noise_cross_correlation + conj(desired_signal(iteration_counter))*current_signal;

    %calculate current least squares solution (after recursively updating Sd & pd, notice that, ...
    %properly, the coefficients themselve are not recursively updated):
    next_coefficients = Sd_inverse_auto_correlation*pd_signal_noise_cross_correlation;

    %keep track of parameters:
    filter_output_over_time(iteration_counter) = current_filter_output;
    filter_error_over_time(iteration_counter) = current_filter_error;
    filter_coefficients_over_time(:,iteration_counter+1) = next_coefficients;
    aposteriori_output_over_time(iteration_counter,1) = filter_coefficients_over_time(:,iteration_counter+1)'*current_signal;
    aposteriori_error_over_time(iteration_counter,1) = desired_signal(iteration_counter)-aposteriori_output_over_time(iteration_counter,1);

end




