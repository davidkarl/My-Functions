function [aposteriori_error_vec,...
          apriori_error_vec,...
          coefficients_vec_over_time] = adaptive_filter_fast_transversal_RLS_stabilized(...
          desired_signal,input_signal,filter_order , lambda_smoothing_factor, initial_aposteriori_error_small)

%   Stab_Fast_RLS.m
%       Implements the Stabilized Fast Transversal RLS algorithm for REAL valued data.
%       (Algorithm 8.2 - book: Adaptive Filtering: Algorithms and Practical
%                                                       Implementation, 3rd Ed., Diniz)
% 
%   Syntax:
%       [posterioriErrorVector,prioriErrorVector,coefficientVector] = Stab_Fast_RLS(desired,input,S)
% 
%   Input Arguments: 
%       . desired           : Desired Signal.                          (ROW Vector)
%       . input             : Signal fed into the adaptive filter.     (ROW Vector)
%       . S                 : Structure with the following fields
%           - lambda             : Forgetting factor.                  (0 << lambda < 1) 
%           - predictorOrder     : refered as N in the textbook.
%           - epsilon            : Initilization of xiMin_backward and xiMin_forward. (usually 0.1 < epsilon <= 1)
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
number_of_coefficients           = filter_order +1;  
number_of_iterations             = length(desired_signal); 

% Pre Allocations
xiMin_f                  = 0;
xiMin_b                  = 0;
gamma_Np1_1              = 0;
gamma_N_2                = 0;
gamma_N_3                = 0;
w_f                      = zeros(number_of_coefficients, 1);
w_b                      = zeros(number_of_coefficients, 1);
coefficients_vec_over_time        = zeros(number_of_coefficients, number_of_iterations+1); 
error_f                  = 0;
error_f_line             = 0;
error_b_line_1           = 0;
error_b_line_2           = 0;
error_b_line_3_Vector    = zeros(3,1);
error_b_3_Vector         = zeros(2,1);
aposteriori_error_vec    = zeros(number_of_iterations,1);
apriori_error_vec        = zeros(number_of_iterations,1);
phiHatN                  = zeros(number_of_coefficients,1);  
phiHatNp1                = zeros(number_of_coefficients+1,1);
regressor                = zeros(number_of_coefficients+1,1);


%%Initialize Parameters
w_f                      = zeros(number_of_coefficients, 1);
w_b                      = zeros(number_of_coefficients, 1);
coefficients_vec_over_time(:,1)   = zeros(number_of_coefficients, 1);
%
phiHatN                  = zeros(number_of_coefficients,1);  
gamma_N_3                = 1;
%
xiMin_f                  = initial_aposteriori_error_small;
xiMin_b                  = initial_aposteriori_error_small;
%
kappa1                  = 1.5;
kappa2                  = 2.5;
kappa3                  = 1;
input_signal = make_column(input_signal);

prefixedInput            = [ zeros(1,number_of_coefficients)  input_signal];

%Loop over the different iterations:
for it = 1:number_of_iterations
    
    regressor    = prefixedInput(it+number_of_coefficients:-1:it).';
    
    error_f_line = regressor.'*[ 1 ;  -w_f];
    error_f      = error_f_line*gamma_N_3;
    phiHatNp1    = [0 ; phiHatN ]  + ...
                   1/(lambda_smoothing_factor*xiMin_f)*[1 ; -w_f]*error_f_line;
    
    % Forward Info 
    gamma_Np1_1  = 1/(1/gamma_N_3  +phiHatNp1(1)*error_f_line );
    xiMin_f      = 1/(1/(xiMin_f*lambda_smoothing_factor) -gamma_Np1_1*(phiHatNp1(1))^2);
    w_f          = w_f  +  phiHatN*error_f;
    
    % Backward Errors
    error_b_line_1           = lambda_smoothing_factor*xiMin_b*phiHatNp1(end);
    error_b_line_2           = [-w_b.' 1]*regressor;
    error_b_line_3_Vector(1) = error_b_line_2*kappa1 + ...
                               error_b_line_1*(1 -kappa1);
    error_b_line_3_Vector(2) = error_b_line_2*kappa2 + ...
                               error_b_line_1*(1 -kappa2);
    error_b_line_3_Vector(3) = error_b_line_2*kappa3 + ...
                               error_b_line_1*(1 -kappa3);
    
    % Backward Coefficients
    gamma_N_2 = 1/(1/gamma_Np1_1 - ...
                   phiHatNp1(end)*error_b_line_3_Vector(3));                         
    error_b_3_Vector(1)      =  error_b_line_3_Vector(1)*gamma_N_2;         
    error_b_3_Vector(2)      =  error_b_line_3_Vector(2)*gamma_N_2;
    xiMin_b   =  lambda_smoothing_factor*xiMin_b + ...
                 error_b_3_Vector(2)*error_b_line_3_Vector(2);
    phiHatN   =  phiHatNp1(1:end-1) +phiHatNp1(end)*w_b;
    w_b       =  w_b   +  phiHatN*error_b_3_Vector(1);        
    gamma_N_3 =  1/(1  + phiHatN.'*regressor(1:end-1));
            
    % Joint Process Estimation
    apriori_error_vec(it)     = desired_signal(it) -...
                                coefficients_vec_over_time(:,it).'*regressor(1:end-1);     
    aposteriori_error_vec(it) = apriori_error_vec(it)*gamma_N_3;                             
    coefficients_vec_over_time(:,it+1) = coefficients_vec_over_time(:,it) + ...
                                phiHatN*aposteriori_error_vec(it);
    

                                
end % END OF LOOP



%EOF

