

%   Tdomain.m
%       Implements the Transform-Domain LMS algorithm for COMPLEX valued data.
%
%   Syntax:
%       [outputVector,errorVector,coefficientVectorT,coefficientVector] = Tdomain(desired,input,S,T)
%
%   Input Arguments:
%       . desired   : Desired signal.                               (ROW vector)
%       . input     : Signal fed into the adaptive filter.          (ROW vector)
%       . S         : Structure with the following fields
%           - step                  : Convergence (relaxation) factor.
%           - filterOrderNo         : Order of the FIR filter.
%           - initialCoefficients   : Initial filter coefficients
%                                     in the ORIGINAL domain.       (COLUMN vector)
%           - gamma                 : Regularization factor.
%                                     (small positive constant to avoid singularity)
%           - alpha                 : Used to estimate eigenvalues of Ru.
%                                     (0 << alpha < 0.1)
%           - initialPower          : Initial power.                (SCALAR)
%       . T         : Transform applied to the signal.  (T must be a unitary MATRIX)
%                                                       (filterOrderNo+1 x filterOrderNo+1)
%
%   Output Arguments:
%       . outputVector      :   Store the estimated output of each iteration.   (COLUMN vector)
%       . errorVector       :   Store the error for each iteration.             (COLUMN vector)
%       . coefficientVectorT:   Store the estimated coefficients for each iteration
%                               in the TRANSFORM domain.
%                               (Coefficients at one iteration are COLUMN vector)
%       . coefficientVector :   Store the estimated coefficients for each iteration
%                               in the ORIGINAL domain.
%                               (Coefficients at one iteration are COLUMN vector)
%
%   Comments:
%       The adaptive filter is implemented in the Transform-Domain. Therefore, the first three
%       output variables are calculated in this TRANSFORMED domain. The last output variable,
%       coefficientVector, corresponds to the adaptive filter coefficients in the ORIGINAL
%       domain (coefficientVector = T' coefficientVectorT) and is only calculated in order to
%       facilitate comparisons, i.e., for implementation purposes just coefficientVectorT
%       matters.
%

function    [filter_output_over_time,filter_error_over_time,filter_coefficients_over_time_in_T_domain,coefficient_vector] = ...
    adaptive_filter_DFT_domain(desired_signal,input_signal,filter_order,initial_coefficients,mu_convergence_factor,gamma_normalization_factor,alpha_smoothing_factor,initial_power)
       
%initialize parameters:
number_of_coefficients = filter_order+1;
N_number_of_iterations = length(desired_signal);

%initialize error, output and coefficients vectors:
filter_error_over_time = zeros(N_number_of_iterations,1);
filter_output_over_time = zeros(N_number_of_iterations,1);
filter_coefficients_over_time_in_T_domain = zeros(number_of_coefficients ,N_number_of_iterations+1);
filter_coefficients_over_time_in_T_domain(:,1) = fft(initial_coefficients)/sqrt(number_of_coefficients);

%initialize auxiliary next coefficients vector and signal power:
next_coefficients_in_T_domain = fft(initial_coefficients)/sqrt(number_of_coefficients);
signal_power_in_T_domain = initial_power*ones(number_of_coefficients,1);

%Improve source code regularity
zero_padded_input_signal = [zeros(number_of_coefficients-1,1) ; input_signal];

%Loop over the different iterations:
for iteration_counter = 1:N_number_of_iterations,
    
    %get input signal:
    current_input_signal = flip(zero_padded_input_signal(iteration_counter:iteration_counter+number_of_coefficients-1));
    
    %transform current signal using T transform:
    current_input_signal_T_transformed = fft(current_input_signal)/sqrt(number_of_coefficients);

    %get current signal power:
    current_signal_power_in_T_transform = current_input_signal_T_transformed .* conj(current_input_signal_T_transformed);
    
    %smooth signal power:
    signal_power_in_T_domain = alpha_smoothing_factor*current_signal_power_in_T_transform+...
                                         (1-alpha_smoothing_factor)*signal_power_in_T_domain;
    
    %get current filter output and error:
    current_coefficients_in_T_domain = next_coefficients_in_T_domain;
    current_filter_output = current_coefficients_in_T_domain' * current_input_signal_T_transformed;
    current_filter_error = desired_signal(iteration_counter) - current_filter_output;
    
    %calculate next iterations coefficients:
    next_coefficients_in_T_domain = current_coefficients_in_T_domain + ...
           mu_convergence_factor * conj(current_filter_error) * current_input_signal_T_transformed ./ (gamma_normalization_factor + signal_power_in_T_domain);
    
    %keep track of parameters:
    filter_output_over_time(iteration_counter) = current_filter_output;
    filter_error_over_time(iteration_counter) = current_filter_error;
    filter_coefficients_over_time_in_T_domain(:,iteration_counter+1) = next_coefficients_in_T_domain;
    
end

coefficient_vector = ifft(filter_coefficients_over_time_in_T_domain)*sqrt(number_of_coefficients);

