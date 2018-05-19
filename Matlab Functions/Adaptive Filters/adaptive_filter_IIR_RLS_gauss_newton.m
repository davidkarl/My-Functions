function [output_signal_final, ...
          filter_error_over_time,...
          theta_total_coefficients] = adaptive_filter_IIR_RLS_gauss_newton(...
          desired_signal,input_signal,numerator_order,denominator_order,mu_step_size,lambda_smoothing_factor,delta_regularization)

%   RLS_IIR.m
%       Implements the RLS version of the Output Error algorithm (also known as RLS adaptive IIR filter) 
%       for REAL valued data.
%       (Algorithm 10.1 - book: Adaptive Filtering: Algorithms and Practical
%                                                        Implementation, 3rd Ed., Diniz)
% 
%   Syntax:
%       [outputVector,errorVector,thetaVector] = RLS_IIR(desired,input,S)
% 
%   Input Arguments: 
%       . desired           : Desired Signal.                          (ROW Vector)
%       . input             : Signal fed into the adaptive filter.     (ROW Vector)
%       . S                 : Structure with the following fields
%           - lambda             : Forgetting factor.                  (0 << lambda < 1) 
%           - M                  : Adaptive filter numerator order, refered as M in the textbook.
%           - N                  : Adaptive filter denominator order, refered as N in the textbook.
%           - delta              : Regularization factor. 
% 
%   Output Arguments:
%       . outputVector      : Store the estimated output for each iteration.  (COLUMN vector)
%       . errorVector       : Store the error for each iteration.             (COLUMN vector)
%       . thetaVector       : Store the estimated coefficients of the IIR filter for each iteration.
%                             (Coefficients at one iteration are COLUMN vector)
%
%   Authors:
%       . Guilherme de Oliveira Pinto   - guilhermepinto7@gmail.com  &  guilherme@lps.ufrj.br
%       . Markus Vinícius Santos Lima   - mvsl20@gmailcom            &  markus@lps.ufrj.br
%       . Wallace Alves Martins         - wallace.wam@gmail.com      &  wallace@lps.ufrj.br
%       . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           &  wagner@lps.ufrj.br
%       . Paulo Sergio Ramirez Diniz    -                               diniz@lps.ufrj.br
%

        

% Initialization Procedure
total_number_of_coefficients = numerator_order + denominator_order + 1;  
number_of_iterations = length(desired_signal);

% Pre Allocations
filter_error_over_time = zeros(number_of_iterations  ,1);
output_signal_final = zeros(number_of_iterations  ,1);
theta_total_coefficients = zeros(total_number_of_coefficients,number_of_iterations+1);
current_signal = zeros(total_number_of_coefficients,1);

% Initial State Weight Vector
Sd_inverse_R = inv(delta_regularization)*eye(total_number_of_coefficients);
x_derivative_vec = zeros(max(numerator_order+1, denominator_order),   1);
y_derivative_vec = zeros(max(numerator_order+1, denominator_order),   1);
PhiVector = zeros(total_number_of_coefficients,1);


% Improve source code regularity (initial state = 0)
x_input_signal = [zeros(numerator_order,1) ; input_signal(:)]; 
y_output_signal = [zeros(denominator_order,1) ; output_signal_final]; 

%Loop over the different iterations:
for iterations_counter = 1:number_of_iterations
    
   current_signal = [ y_output_signal(iterations_counter+(denominator_order-1):-1:iterations_counter) ; ...
                       x_input_signal(iterations_counter+numerator_order:-1:iterations_counter)];
                              
   %Compute Output                         
   y_output_signal(iterations_counter+denominator_order) = ...
                                        theta_total_coefficients(:,iterations_counter).' * current_signal;

   %Error:
   filter_error_over_time(iterations_counter) = desired_signal(iterations_counter) - ...
                                                    y_output_signal(iterations_counter+denominator_order);    
   
   %Xline,Yline
   denominator_coefficients = theta_total_coefficients(1:denominator_order,iterations_counter);
   
   x_derivative_approximation = x_input_signal(iterations_counter+numerator_order) + ...
                                        denominator_coefficients.' * x_derivative_vec(1:denominator_order); 
   
   y_derivative_approximation = -y_output_signal(iterations_counter+denominator_order-1) + ...
                                        denominator_coefficients.' * y_derivative_vec(1:denominator_order); 
   x_derivative_vec = [ x_derivative_approximation ; x_derivative_vec(1:end-1) ];
   y_derivative_vec = [ y_derivative_approximation ; y_derivative_vec(1:end-1) ];
   PhiVector   = [ +y_derivative_vec(1:denominator_order) ; -x_derivative_vec(1:numerator_order+1) ];
         
   
   % Sd
   Sd_inverse_R = inv(lambda_smoothing_factor)*...
               (Sd_inverse_R - (Sd_inverse_R*(PhiVector*PhiVector.')*Sd_inverse_R)/...
               (lambda_smoothing_factor/(1-lambda_smoothing_factor) + PhiVector.'*Sd_inverse_R*PhiVector));
            
   
    % Update Coefficients
    theta_total_coefficients(:,iterations_counter+1) = ...
                    theta_total_coefficients(:,iterations_counter) - ...
                       mu_step_size * Sd_inverse_R * PhiVector * filter_error_over_time(iterations_counter);                      
    
                                
    %Stability Procedure
    theta_total_coefficients(1:denominator_order, iterations_counter+1) = ...
                stabilize_coefficients(theta_total_coefficients(1:denominator_order,iterations_counter+1));
                  
end



output_signal_final = y_output_signal(denominator_order+1:end);



%
% Sub function
%
%


function [coeffOut] = stabilize_coefficients(coeffIn)
   

poles                = roots([1 -coeffIn.']);

indexVector          = find(abs(poles) > 1);
%poles(indexVector)   = poles(indexVector)./(abs(poles(indexVector)).^2);
poles(indexVector)   = 1./poles(indexVector);
coeffOut             = poly(poles);

coeffOut              = -real(coeffOut(2:end)).';


%EOF


