function [ladder_vector,...
          kappa_vector,...
          aposteriori_error_matrix] = adaptive_filter_lattice_NRLS_aposteriori_errors(...
            desired_signal,input_signal,filter_order, lambda_smoothing_factor, initial_aposteriori_error_small)

%   NLRLS_pos.m
%       Implements the Normalized Lattice RLS algorithm based on a posteriori error.
%       (Algorithm 7.2 - book: Adaptive Filtering: Algorithms and Practical
%                                                       Implementation, Diniz)
% 
%   Syntax:
%       [ladderVector,kappaVector,posterioriErrorMatrix] = NLRLS_pos(desired,input,S)
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

%Basic sizes:
number_of_coefficients = filter_order + 1;  
number_of_iterations = length(desired_signal); 

%Initialize variables:
delta = zeros(1, filter_order + 1);
delta_D = zeros(1, filter_order + 1);
error_b_current = zeros(1, filter_order + 2);
error_b_previous = zeros(1, filter_order + 2);
error_f = zeros(1, filter_order + 2);
ladder_vector = zeros(number_of_coefficients , number_of_iterations); 
kappa_vector = zeros(number_of_coefficients , number_of_iterations); 
aposteriori_error_matrix = zeros(filter_order + 2, number_of_iterations);

%%Initialize Xi_min not with zero but with small initial aposteriori error:
sigma_d_2 = initial_aposteriori_error_small;
sigma_x_2 = initial_aposteriori_error_small;


%Loop over the different iterations:
for iterations_counter = 1:number_of_iterations

    sigma_x_2 = lambda_smoothing_factor*sigma_x_2 + input_signal(iterations_counter)^2;
    sigma_d_2 = lambda_smoothing_factor*sigma_d_2 + desired_signal(iterations_counter)^2;
    
    %Set Values for Section 0(Zero):
    error_b_current(1) = input_signal(iterations_counter)/(sigma_x_2^0.5);
    error_f(1) = error_b_current(1);
    aposteriori_error_matrix(1,iterations_counter) = desired_signal(iterations_counter)/(sigma_d_2^0.5);
    
    % Propagate the Order Update Equations:
    for order_counter = 1:filter_order+1
    
        %Delta Time Update
        delta(order_counter) = delta(order_counter) *...
           sqrt((1 - error_b_previous(order_counter)^2) * (1 - error_f(order_counter)^2)) ...
                                + error_b_previous(order_counter)*error_f(order_counter);

        %Order Update Equations                        
        error_b_current(order_counter+1) = ...
            (error_b_previous(order_counter) - delta(order_counter)*error_f(order_counter))...
                        / sqrt((1 -delta(order_counter)^2)*(1 -error_f(order_counter)^2));
                          
        error_f(order_counter+1) = ...
            (error_f(order_counter) - delta(order_counter)*error_b_previous(order_counter))...
                        / sqrt((1 -delta(order_counter)^2)*(1 -error_b_previous(order_counter)^2));
 
        %Feedforward Filtering:    
        delta_D(order_counter) = delta_D(order_counter) *...
            sqrt((1 - error_b_current(order_counter)^2) * (1 - aposteriori_error_matrix(order_counter,iterations_counter)^2))...
                    + aposteriori_error_matrix(order_counter,iterations_counter)*error_b_current(order_counter);
        
        
        aposteriori_error_matrix(order_counter+1,iterations_counter) = ...
            1/sqrt((1 - error_b_current(order_counter)^2)*(1 -delta_D(order_counter)^2)) *...
            (aposteriori_error_matrix(order_counter,iterations_counter) ...
                - delta_D(order_counter)*error_b_current(order_counter));

         %Outputs:
         ladder_vector(order_counter,iterations_counter) = delta_D(order_counter);
         kappa_vector(order_counter,iterations_counter) = delta(order_counter);
 
    end
    
    %Upadate K-1 Info:
    error_b_previous = error_b_current;


end % END OF LOOP

%EOF

