function [step_response] = get_AR_parameters_step_response(AR_parameters,step_response_length)
% Calculate the step response of an AR system described by a coefficients
% Returns vector of length n
% Usage: imp=a2step(a,n)

step_response = simulate_ARMAX_system(AR_parameters,1,0,ones(1,step_response_length)) ;

