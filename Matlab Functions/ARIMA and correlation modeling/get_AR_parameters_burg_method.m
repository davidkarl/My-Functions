function  [AR_parameters,forward_prediction_error,reflection_coefficient]=get_AR_parameters_burg_method(auto_correlation_sequence,AR_model_order);

% Function BURG: Computation of AR model using Burg's algorithm (forward and backward error joint minimization)
% (symmetrical lattice filter i think, go over it again)
% Usage: [a,alpha,rc]=burg(x,p);
% Input parameters:
%   x    - processing frame
%   p    - order of LPC model
% Output parameters:
%   a     - autoregressive coefficients 
%   alpha - forward prediction error
%   rc    - reflection coefficients
%

%  changed by JK, 11.9.1997 to avoid division by zero 
%  changed output format (skipped first coefficient and changed sign)

auto_correlation_sequence = auto_correlation_sequence(:);
auto_correlation_length = length(auto_correlation_sequence);

%initialize forward and backward prediction error:
forward_error = auto_correlation_sequence;
backward_error = auto_correlation_sequence;

%initialize forward (aposteriori) prediction error and loop counter:
forward_prediction_error = sum(forward_error.*forward_error)/auto_correlation_length;
AR_calculation_loop_counter=1;

%initialize denominator and numinator for calculation: 
den = sum( forward_error(2:auto_correlation_length) .* forward_error(2:auto_correlation_length) )...
    + sum( backward_error(1:auto_correlation_length-1) .* backward_error(1:auto_correlation_length-1) );
        
num = 2*sum( forward_error(2:auto_correlation_length) .* backward_error(1:auto_correlation_length-1) );


%here we might have divided by zero:
if den>eps,
    reflection_coefficient(1) =  - num / den;
    AR_parameters(1) = - num / den ;
else
    reflection_coefficient(1) = 0 ; 
    AR_parameters(1) = 0 ;
end 

%initialize previous forward and backward prediction errors:
forward_error_previous = forward_error;
backward_error_previous = backward_error;
forward_error(2:auto_correlation_length) = forward_error(2:auto_correlation_length) + ...
    reflection_coefficient(AR_calculation_loop_counter)*backward_error_previous(1:auto_correlation_length-1);

backward_error(2:auto_correlation_length) = backward_error_previous(1:auto_correlation_length-1) + ...
    reflection_coefficient(AR_calculation_loop_counter)*forward_error_previous(2:auto_correlation_length);

forward_prediction_error = forward_prediction_error * (1-reflection_coefficient(AR_calculation_loop_counter)^2);

AR_parameters_previous = AR_parameters;

%start burg algorithm loop:
for AR_calculation_loop_counter=2:AR_model_order,
    
    %calculate numinator and denominator:
    den = sum(forward_error(1+AR_calculation_loop_counter:auto_correlation_length).*forward_error(1+AR_calculation_loop_counter:auto_correlation_length)) ...
        + sum(backward_error(AR_calculation_loop_counter:auto_correlation_length-1).*backward_error(AR_calculation_loop_counter:auto_correlation_length-1));
    
    num = 2*sum(forward_error(1+AR_calculation_loop_counter:auto_correlation_length).*backward_error(AR_calculation_loop_counter:auto_correlation_length-1));
    
    %get reflection coefficient and current AR parameter calculated unless
    %denominator is too close to zero, in which case zero them both:
    if den>eps,
        reflection_coefficient(AR_calculation_loop_counter) =  - num / den;
        AR_parameters(AR_calculation_loop_counter) = - num / den ;
    else
        reflection_coefficient(AR_calculation_loop_counter)=0 ; 
        AR_parameters(AR_calculation_loop_counter)=0 ;
    end
    
    %update forward and backward prediction errors:
    forward_error_previous = forward_error;
    backward_error_previous = backward_error;
    
    forward_error(2:auto_correlation_length) = forward_error(2:auto_correlation_length) + ...
        reflection_coefficient(AR_calculation_loop_counter)*backward_error_previous(1:auto_correlation_length-1);
    
    backward_error(2:auto_correlation_length) = backward_error_previous(1:auto_correlation_length-1) + ...
        reflection_coefficient(AR_calculation_loop_counter)*forward_error_previous(2:auto_correlation_length);
    
    forward_prediction_error = forward_prediction_error*(1-reflection_coefficient(AR_calculation_loop_counter)^2);
    AR_parameters(1:AR_calculation_loop_counter-1) = AR_parameters_previous(1:AR_calculation_loop_counter-1) ...
        + reflection_coefficient(AR_calculation_loop_counter)*fliplr(AR_parameters_previous(1:AR_calculation_loop_counter-1));
    
    %update previous AR parameters variable with current AR paramters:
    AR_parameters_previous = AR_parameters;
end;

%output parameters i the common form by adding a minus sign:
AR_parameters = -AR_parameters;




