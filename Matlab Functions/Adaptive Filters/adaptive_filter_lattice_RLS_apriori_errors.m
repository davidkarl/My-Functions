function [ladder_vector,...
          kappa_vector,...
          apriori_error_matrix] = adaptive_filter_lattice_RLS_apriori_errors(...
            desired_signal,input_signal, filter_order, lambda_smoothing_factor, initial_aposteriori_error_small)

%   LRLS_priori.m
%       Implements the Lattice RLS algorithm based on a priori errors.
%       (Algorithm 7.4 - book: Adaptive Filtering: Algorithms and Practical
%                                                       Implementation, Diniz)
% 
%   Syntax:
%       [ladderVector,kappaVector,prioriErrorMatrix] = LRLS_priori(desired,input,S)
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
%       . prioriErrorMatrix     : Store the a priori errors at each section of the lattice for 
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

%Basic sizes:
number_of_coefficients = filter_order +1;  
number_of_iterations = length(desired_signal); 

%Initialize variables:
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
apriori_error_matrix = zeros(filter_order +2, number_of_iterations);
kappa_f = zeros(1, filter_order +1);
kappa_b = zeros(1, filter_order +1);


%%Initialize Xi_min not with zero but with small initial aposteriori error:
xi_min_f = repmat(initial_aposteriori_error_small, 1 , filter_order +2);
xi_min_b_previous = repmat(initial_aposteriori_error_small , 1, filter_order +2);
ladder_vector(:,1) = zeros(number_of_coefficients, 1);


%Loop over the different iterations:
for iterations_counter = 1:number_of_iterations

    % Set Values for Section 0(Zero)
    gamma_current(1) = 1;
    error_b_current(1) = input_signal(iterations_counter);
    error_f(1) = input_signal(iterations_counter);
    xi_min_f(1) = input_signal(iterations_counter)^2 + lambda_smoothing_factor*xi_min_f(1);
    xi_min_b_current(1) = xi_min_f(1);
    apriori_error_matrix(1,iterations_counter) = desired_signal(iterations_counter);
    
    % Propagate the Order Update Equations
    for order_counter = 1:filter_order+1
    
        %Delta Time Update
        delta(order_counter) = lambda_smoothing_factor*delta(order_counter) + ...
                     gamma_previous(order_counter)*error_b_previous(order_counter)*error_f(order_counter);
                      
        %Order Update Equations                        
        gamma_current(order_counter+1) = gamma_current(order_counter) - ...
            ((gamma_current(order_counter)*error_b_current(order_counter))^2)/xi_min_b_current(order_counter);
        
        error_b_current(order_counter+1) = error_b_previous(order_counter) - kappa_b(order_counter)*error_f(order_counter);
        error_f(order_counter+1) = error_f(order_counter) - kappa_f(order_counter)*error_b_previous(order_counter);

        kappa_f(order_counter) = delta(order_counter)/xi_min_b_previous(order_counter);
        kappa_b(order_counter) = delta(order_counter)/xi_min_f(order_counter);
        
        xi_min_f(order_counter+1) = xi_min_f(order_counter) - delta(order_counter)*kappa_f(order_counter);
        xi_min_b_current(order_counter+1) = xi_min_b_previous(order_counter) - delta(order_counter)*kappa_b(order_counter);
        
        %Feedforward Filtering:
        delta_D(order_counter) = lambda_smoothing_factor*delta_D(order_counter) + ...
                            gamma_current(order_counter)*error_b_current(order_counter)*...
                            apriori_error_matrix(order_counter,iterations_counter);
        
        apriori_error_matrix(order_counter+1,iterations_counter) = ...
            apriori_error_matrix(order_counter,iterations_counter) - ...
                      ladder_vector(order_counter,iterations_counter)*error_b_current(order_counter);

        ladder_vector(order_counter,iterations_counter+1) = delta_D(order_counter)/xi_min_b_current(order_counter);
        
        kappa_vector(order_counter,iterations_counter) = kappa_f(order_counter);
 
    end

    
    %Upadate K-1 Info
    gamma_previous = gamma_current;
    error_b_previous     = error_b_current;
    xi_min_b_previous     = xi_min_b_current;


end % END OF LOOP
1;
%EOF

