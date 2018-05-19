function [impulse_response] = get_AR_parameters_impulse_response(AR_parameters,impulse_response_length)
% Calculate the impulse response of an AR system described by a coefficients
% Returns vector of length n
% Usage: imp=a2imp(a,n)

impulse_response = simulate_ARMAX_system(AR_parameters,1,0,[1 zeros(1,impulse_response_length-1)]);


