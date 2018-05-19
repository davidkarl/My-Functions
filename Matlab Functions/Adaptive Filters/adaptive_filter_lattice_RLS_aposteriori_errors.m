function [ladder_vector,...
          kappa_vector,...
          aposteriori_error_matrix] = adaptive_filter_lattice_RLS_aposteriori_errors(...
                      desired_signal, input_signal, filter_order, lambda_smoothing_factor, initial_aposteriori_error_small)

%   LRLS_pos.m
%       Implements the Lattice RLS algorithm based on a posteriori errors.
%       (Algorithm 7.1 - book: Adaptive Filtering: Algorithms and Practical
%                                                       Implementation, Diniz)
% 
%   Syntax:
%       [ladderVector,kappaVector,posterioriErrorMatrix] = LRLS_pos(desired,input,S)
% 
%   Input Arguments: 
%       . desired           : Desired Signal.                          (ROW Vector)
%       . input             : Signal fed into the adaptive filter.     (ROW Vector)
%       . S                 : Structure with the following fields
%           - lambda             : Forgetting factor.                  (0 << lambda < 1)
%           - nSectionsLattice   : Number of lattice sections, refered as N in the textbook.
%           - epsilon            : Initilization of xiMin_backward and xiMin_forward. 
% 
%   Output Arguments:
%       . ladderVector          : Store the ladder coefficients for each iteration, refered 
%                                 as v in the textbook.
%                                 (Coefficients at one iteration are column vector)
%       . kappaVector           : Store the reflection coefficients for each iteration, considering 
%                                 only the forward part.
%                                 (Coefficients at one iteration are column vector)
%       . posterioriErrorMatrix : Store the a posteriori errors at each section of the lattice for 
%                                 each iteration.
%                                 (The errors at one iteration are column vectors)
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

% Basic Procedure
number_of_coefficients = filter_order +1;  
number_of_iterations = length(desired_signal); 

%Iniialize variables:
delta = zeros(1, filter_order +1);
delta_D = zeros(1, filter_order +1);
xi_min_b_current = zeros(1, filter_order +2);
gamma_current = zeros(1, filter_order +2);
gamma_previous = ones(1, filter_order +2);
error_b_current = zeros(1, filter_order +2);
error_b_previous = zeros(1, filter_order +2);
error_f = zeros(1, filter_order +2);
ladder_vector = zeros(number_of_coefficients , number_of_iterations); 
kappa_vector = zeros(number_of_coefficients , number_of_iterations); 
aposteriori_error_matrix = zeros(filter_order + 2 , number_of_iterations);
kappa_f = 0;
kappa_b = 0;

%%Initialize Xi_min not with zero but with small initial aposteriori error:
xi_min_f = repmat(initial_aposteriori_error_small,1, filter_order +2);
xi_min_b_previous = repmat(initial_aposteriori_error_small,1, filter_order +2);



%Loop over the different iterations:
for iteration_counter = 1:number_of_iterations

    %Initialize Values for Section 0(Zero)
    gamma_current(1) = 1;
    error_b_current(1) = input_signal(iteration_counter);
    error_f(1) = input_signal(iteration_counter);
    xi_min_f(1) = input_signal(iteration_counter)^2 + lambda_smoothing_factor*xi_min_f(1);
    xi_min_b_current(1) = xi_min_f(1);
    aposteriori_error_matrix(1,iteration_counter) = desired_signal(iteration_counter);
    
    % Propagate the Order Update Equations
    for order_counter = 1:filter_order+1
    
        %Delta Time Update
        delta(order_counter) = lambda_smoothing_factor*delta(order_counter) + ...
              (error_b_previous(order_counter)*error_f(order_counter))/gamma_previous(order_counter);
                      
        %Order Update Equations                        
        gamma_current(order_counter+1) = gamma_current(order_counter) - ...
                            (error_b_current(order_counter)^2)/xi_min_b_current(order_counter);
        
        kappa_b = delta(order_counter)/xi_min_f(order_counter);
        kappa_f = delta(order_counter)/xi_min_b_previous(order_counter);
        
        error_b_current(order_counter+1) = error_b_previous(order_counter) - kappa_b*error_f(order_counter);
        error_f(order_counter+1) = error_f(order_counter) - kappa_f*error_b_previous(order_counter);
        
        xi_min_b_current(order_counter+1) = ...
                        xi_min_b_previous(order_counter) - delta(order_counter)*kappa_b; 
        xi_min_f(order_counter+1) = ...
                        xi_min_f(order_counter) - delta(order_counter)*kappa_f;
        
        %Feedforward Filtering:    
        delta_D(order_counter) = lambda_smoothing_factor*delta_D(order_counter) + ...
                            (error_b_current(order_counter)/gamma_current(order_counter))*...
                                    aposteriori_error_matrix(order_counter,iteration_counter);
        
        ladder_vector(order_counter,iteration_counter) = ...
                                    delta_D(order_counter) / xi_min_b_current(order_counter);
        
        aposteriori_error_matrix(order_counter+1,iteration_counter) = ...
                    aposteriori_error_matrix(order_counter,iteration_counter) - ...
                          ladder_vector(order_counter,iteration_counter)*error_b_current(order_counter);

        kappa_vector(order_counter,iteration_counter) = kappa_f;
 
    end %end of order update counter

    
    %Upadate K-1 Info:
    gamma_previous = gamma_current;
    error_b_previous = error_b_current;
    xi_min_b_previous = xi_min_b_current;


end % END OF MAIN ITERATIONS LOOP

%EOF
