function [filter_output_input, ...
          filter_desired_output_error,...
          theta_total_coefficients,...
          filter_desired_input_error] = adaptive_filter_IIR_RLS_error_equations(...
            desired_signal,input_signal,numerator_order,denominator_order,lambda_smoothing_factor,delta_regularization)

%   ErrorEquation.m
%       Implements the Error Equation RLS algorithm for REAL valued data.
%       (Algorithm 10.3 - book: Adaptive Filtering: Algorithms and Practical
%                                                        Implementation, 3rd Ed., Diniz)
% 
%   Syntax:
%       [outputVector,errorVector,thetaVector] = ErrorEquation(desired,input,S)
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
%       . outputVector      : Store the estimated output for each iteration.           (COLUMN vector)
%       . errorVector       : Store the error for each iteration.                      (COLUMN vector)
%       . thetaVector       : Store the estimated coefficients of the IIR filter for each iteration.
%                             (Coefficients at one iteration are COLUMN vector)
%       . errorVector_e     : Store the auxiliary error used for updating thetaVector  (COLUMN vector)
%
%   Authors:
%       . Guilherme de Oliveira Pinto   - guilhermepinto7@gmail.com  &  guilherme@lps.ufrj.br
%       . Markus Vinícius Santos Lima   - mvsl20@gmailcom            &  markus@lps.ufrj.br
%       . Wallace Alves Martins         - wallace.wam@gmail.com      &  wallace@lps.ufrj.br
%       . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           &  wagner@lps.ufrj.br
%       . Paulo Sergio Ramirez Diniz    -                               diniz@lps.ufrj.br
%


%Basic sizes:
total_number_of_coefficients = numerator_order + denominator_order + 1;  
number_of_iterations = length(desired_signal);

%Initialize variables:
filter_desired_output_error = zeros(number_of_iterations  ,1);
filter_desired_input_error = zeros(number_of_iterations  ,1);
filter_output_input = zeros(number_of_iterations  ,1);
filter_desired_input = zeros(number_of_iterations  ,1);
theta_total_coefficients = zeros(total_number_of_coefficients,number_of_iterations+1);
current_output_input_signl = zeros(total_number_of_coefficients,1);
current_desired_input_signal = zeros(total_number_of_coefficients,1);
Sd_inverse_R = inv(delta_regularization)*eye(total_number_of_coefficients);
input_signal = [zeros(numerator_order,1) ; input_signal(:)]; 
filter_output_input = [zeros(denominator_order,1) ; filter_output_input]; 
zero_padded_desired_signal = [zeros(denominator_order,1) ; desired_signal(:)]; 


%Loop over the different iterations:
for iterations_counter = 1:number_of_iterations
    
   current_output_input_signl = [ filter_output_input(iterations_counter+(denominator_order-1):-1:iterations_counter)... 
                                  ; input_signal(iterations_counter+numerator_order:-1:iterations_counter)];
                            
   current_desired_input_signal = [ zero_padded_desired_signal(iterations_counter+(denominator_order-1):-1:iterations_counter)
                                input_signal(iterations_counter+numerator_order:-1:iterations_counter)];
                              
   %Compute Output:                         
   filter_output_input(iterations_counter+denominator_order) = ...
                            theta_total_coefficients(:,iterations_counter).' * current_output_input_signl;
   filter_desired_input(iterations_counter+denominator_order) = ...
                            theta_total_coefficients(:,iterations_counter).' * current_desired_input_signal;
                        
   %Error:
   filter_desired_output_error(iterations_counter) = ...
            desired_signal(iterations_counter) - filter_output_input(iterations_counter+denominator_order);
   filter_desired_input_error(iterations_counter) = ...
            desired_signal(iterations_counter) - filter_desired_input(iterations_counter+denominator_order);        
   
   %Sd:
   Sd_inverse_R = inv(lambda_smoothing_factor)*...
   (Sd_inverse_R - (Sd_inverse_R*(current_desired_input_signal*current_desired_input_signal.')*Sd_inverse_R)/...
      (lambda_smoothing_factor + current_desired_input_signal.'*Sd_inverse_R*current_desired_input_signal));
            
   
    % Update Coefficients
    theta_total_coefficients(:,iterations_counter+1) = ...
        theta_total_coefficients(:,iterations_counter) + ...
               Sd_inverse_R * current_desired_input_signal * filter_desired_input_error(iterations_counter);                      
    
                                
    %Stability Procedure
    theta_total_coefficients(1:denominator_order, iterations_counter+1) = ...
                stabilize_coefficients(theta_total_coefficients(1:denominator_order,iterations_counter+1));
  
                  
end



filter_output_input = filter_output_input(denominator_order+1:end);



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


