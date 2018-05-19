function [AR_parameters] = get_AR_parameters(input_signal,AR_model_order) 
% Given a signal x,  this function computes the 'a'
% parameters of an AR synthetising filter of the order n
% It uses normal equations derived from Cayley-Hamilton theorem
% Usage: a=x2a(x,n)

% The length of the input vector
input_signal_length = length(input_signal);
if (2*AR_model_order>=input_signal_length) 
    error('The signal is too short.'); 
end

%Here corrn can be replaced by corrs
auto_correlation_sequence = get_auto_correlation_using_direct_calculation(input_signal,2*AR_model_order+1) ;

%Calculate and stabilize AR parameters (look at the difference in results
%between difference methods and don't foget matlab's ar() function:
AR_parameters = auto_correlation_to_AR_parameters_using_CH_theorem(auto_correlation_sequence,AR_model_order) ;
AR_parameters = stabilize_AR_parameters_by_moving_them_inside_the_unit_circle(AR_parameters) ;









