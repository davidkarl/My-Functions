function [filter_output_signal, ...
          filter_output_error,...
          theta_total_coefficients,...
          errorVector_s] = adaptive_filter_IIR_Steiglitz_McBride(...
            desired_signal,input_signal,numerator_order,denominator_order,mu_step_size)

%   Steiglitz_McBride.m
%       Implements the Error Equation RLS algorithm for REAL valued data.
%       (Algorithm 10.4 - book: Adaptive Filtering: Algorithms and Practical
%                                                        Implementation, 3rd Ed., Diniz)
% 
%   Syntax:
%       [outputVector,errorVector,thetaVector] = Steiglitz_McBride(desired,input,S)
% 
%   Input Arguments: 
%       . desired           : Desired Signal.                          (ROW Vector)
%       . input             : Signal fed into the adaptive filter.     (ROW Vector)
%       . S                 : Structure with the following fields
%           - step               : Step-size.
%           - M                  : Adaptive filter numerator order, refered as M in the textbook.
%           - N                  : Adaptive filter denominator order, refered as N in the textbook.
% 
%   Output Arguments:
%       . outputVector      : Store the estimated output for each iteration.           (COLUMN vector)
%       . errorVector       : Store the error for each iteration.                      (COLUMN vector)
%       . thetaVector       : Store the estimated coefficients of the IIR filter for each iteration.
%                             (Coefficients at one iteration are COLUMN vector)
%       . errorVector_s     : Store the auxiliary error used for updating thetaVector  (COLUMN vector)
%
%   Authors:
%       . Guilherme de Oliveira Pinto   - guilhermepinto7@gmail.com  &  guilherme@lps.ufrj.br
%       . Markus Vinícius Santos Lima   - mvsl20@gmailcom            &  markus@lps.ufrj.br
%       . Wallace Alves Martins         - wallace.wam@gmail.com      &  wallace@lps.ufrj.br
%       . Luiz Wagner Pereira Biscainho - cpneqs@gmail.com           &  wagner@lps.ufrj.br
%       . Paulo Sergio Ramirez Diniz    -                               diniz@lps.ufrj.br
%

        

%Basic sizes:
number_of_coefficients = numerator_order + denominator_order + 1;  
number_of_iterations = length(desired_signal);

%Initialization:
filter_output_error = zeros(number_of_iterations , 1);
errorVector_s = zeros(number_of_iterations , 1);
filter_output_signal = zeros(number_of_iterations , 1);
theta_total_coefficients = zeros(number_of_coefficients , number_of_iterations+1);
current_output_input_signal = zeros(number_of_coefficients , 1);
regressor_s = zeros(number_of_coefficients , 1);


%Initial State Weight Vector
x_f = zeros(max(numerator_order+1, denominator_order),   1);
d_f = zeros(max(numerator_order+1, denominator_order),   1);

%Improve source code regularity (initial state = 0)
input_signal = [zeros(numerator_order,1) ; input_signal(:)]; 
filter_output_signal = [zeros(denominator_order,1) ; filter_output_signal]; 

%Loop over the different iterations:
for iterations_counter = 1:number_of_iterations
    
   current_output_input_signal = [ filter_output_signal(iterations_counter+(denominator_order-1):-1:iterations_counter)
                                input_signal(iterations_counter+numerator_order:-1:iterations_counter)];
                              
   %Compute Output:                         
   filter_output_signal(iterations_counter+denominator_order) = ...
                theta_total_coefficients(:,iterations_counter).' * current_output_input_signal;

   %Error:
   filter_output_error(iterations_counter) = ...
                desired_signal(iterations_counter) - filter_output_signal(iterations_counter+denominator_order);
   
   %Xline,Yline:
   x_fK = input_signal(iterations_counter+numerator_order) + ...
              theta_total_coefficients(1:denominator_order,iterations_counter).' * x_f(1:denominator_order); 
   
   d_fK = desired_signal(iterations_counter) + ...
              theta_total_coefficients(1:denominator_order,iterations_counter).' * d_f(1:denominator_order); 
  
   x_f = [ x_fK ; x_f(1:end-1) ];
   d_f = [ d_fK ; d_f(1:end-1) ];
   
   regressor_s = [ +d_f(1:denominator_order) ; +x_f(1:numerator_order+1) ];

   %Auxiliar Error:
   errorVector_s(iterations_counter) = d_fK - theta_total_coefficients(:,iterations_counter).'*regressor_s;
                     
    %Update Coefficients:
    theta_total_coefficients(:,iterations_counter+1) = ...
        theta_total_coefficients(:,iterations_counter) + 2*mu_step_size*regressor_s*filter_output_error(iterations_counter);                      
    
                                
    %Stability Procedure
    theta_total_coefficients(1:denominator_order, iterations_counter+1) = stabilityProcedure(theta_total_coefficients(1:denominator_order,iterations_counter+1));
  
                  
end



filter_output_signal = filter_output_signal(denominator_order+1:end);



%
% Sub function
%
%


function [coeffOut] = stabilityProcedure(coeffIn)
   

poles                = roots([1 -coeffIn.']);

indexVector          = find(abs(poles) > 1);
%poles(indexVector)   = poles(indexVector)./(abs(poles(indexVector)).^2);
poles(indexVector)   = 1./poles(indexVector);
coeffOut             = poly(poles);

coeffOut              = -real(coeffOut(2:end)).';


%EOF


