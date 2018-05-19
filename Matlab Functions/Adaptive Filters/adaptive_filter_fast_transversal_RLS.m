function [aposteriori_error_vec,...
          apriori_error_vec,...
          coefficient_vec_over_time] = adaptive_filter_fast_transversal_RLS(...
          desired_signal,input_signal,filter_order , lambda_smoothing_factor, initial_aposteriori_error_small)

%   Fast_RLS.m
%       Implements the Fast Transversal RLS algorithm for REAL valued data.
%       (Algorithm 8.1 - book: Adaptive Filtering: Algorithms and Practical
%                                                       Implementation, 3rd Ed., Diniz)
% 
%   Syntax:
%       [posterioriErrorVector,prioriErrorVector,coefficientVector] = Fast_RLS(desired,input,S)
% 
%   Input Arguments: 
%       . desired           : Desired Signal.                          (ROW Vector)
%       . input             : Signal fed into the adaptive filter.     (ROW Vector)
%       . S                 : Structure with the following fields
%           - lambda             : Forgetting factor.                  (0 << lambda < 1)  
%           - predictorOrder     : refered as N in the textbook.
%           - epsilon            : Initilization of xiMin_backward and xiMin_forward. (usually 0 < epsilon <= 1)
% 
%   Output Arguments:
%       . posterioriErrorVector  : Store the a posteriori error for each iteration.      (COLUMN vector)
%       . prioriErrorVector      : Store the a priori error for each iteration.          (COLUMN vector)
%       . coefficientVector      : Store the estimated coefficients for each iteration.
%                                  (Coefficients at one iteration are COLUMN vector)
%
%   Authors:
%       . Guilherme de Oliveira Pinto   - guilhermepinto7@gmail.com  &  guilherme@lps.ufrj.br
%       . Markus Vinícius Santos Lima   - mvsl20@gmailcom            &  markus@lps.ufrj.br
%       . Wallace Alves Martins         - wallace.wam@gmail.com      &  wallace@lps.ufrj.br
%       . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           &  wagner@lps.ufrj.br
%       . Paulo Sergio Ramirez Diniz    -                               diniz@lps.ufrj.br
%



%################################################
% Data Initialization
%################################################

% Basic Parameters
number_of_coefficients = filter_order + 1;  
number_of_iterations = length(desired_signal); 

% Pre Allocations
xi_min_f_current = 0;
gamma_N_plus_1 = 0;
gamma_N = 1;
w_f = zeros(number_of_coefficients, 1);
w_b = zeros(number_of_coefficients, 1);
coefficient_vec_over_time = zeros(number_of_coefficients, number_of_iterations+1); 
error_f_aposteriori = 0;
error_f_apriori = 0;
error_b_aposteriori = 0;
error_b_apriori = 0;
aposteriori_error_vec = zeros(number_of_iterations,1);
apriori_error_vec = zeros(number_of_iterations,1);
phiHatN = zeros(number_of_coefficients,1);  
phiHatN_plus_1 = zeros(number_of_coefficients+1,1);
current_signal = zeros(number_of_coefficients+1,1);

%Initialize Parameters:
coefficient_vec_over_time(:,1) = zeros(number_of_coefficients, 1); 
xi_min_f_previous = initial_aposteriori_error_small;
xi_min_b = initial_aposteriori_error_small;
input_signal = make_column(input_signal);
zero_padded_input_signal = [ zeros(1,number_of_coefficients) , input_signal];
 

%Loop over the different iterations:
for iterations_counter = 1:number_of_iterations

    current_signal = zero_padded_input_signal(iterations_counter+number_of_coefficients:-1:iterations_counter).';     
    
    
    error_f_apriori = current_signal.' * [ 1 ; -w_f];
    error_f_aposteriori  = error_f_apriori * gamma_N;
    xi_min_f_current = lambda_smoothing_factor*xi_min_f_previous + error_f_apriori*error_f_aposteriori;    
    w_f = w_f + phiHatN*error_f_aposteriori;
    
    phiHatN_plus_1 = [0 ; phiHatN ] + 1/(lambda_smoothing_factor*xi_min_f_previous)*[1 ; -w_f]*error_f_apriori;
    error_b_apriori = lambda_smoothing_factor * xi_min_b * phiHatN_plus_1(end);
    gamma_N_plus_1 = (lambda_smoothing_factor*xi_min_f_previous*gamma_N) / xi_min_f_current;
    gamma_N  = 1/(1/gamma_N_plus_1 - phiHatN_plus_1(end)*error_b_apriori);
    
    error_b_aposteriori = error_b_apriori * gamma_N;
    xi_min_b  = lambda_smoothing_factor*xi_min_b + error_b_aposteriori*error_b_apriori;
    phiHatN = phiHatN_plus_1(1:end-1) + phiHatN_plus_1(end)*w_b; 
    w_b = w_b + phiHatN*error_b_aposteriori;  
    
    
    % Joint Process Estimation
    apriori_error_vec(iterations_counter) = desired_signal(iterations_counter) -...
                                coefficient_vec_over_time(:,iterations_counter).' * current_signal(1:end-1);
    aposteriori_error_vec(iterations_counter) = apriori_error_vec(iterations_counter)*gamma_N;
    coefficient_vec_over_time(:,iterations_counter+1) = coefficient_vec_over_time(:,iterations_counter) + ...
                                phiHatN*aposteriori_error_vec(iterations_counter);
    
    % K- 1 Info
    xi_min_f_previous = xi_min_f_current;


end % END OF LOOP


%EOF

