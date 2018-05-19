function [state_transition_matrix] = AR_parameters_to_state_transfer_matrix(AR_parameters)
% ar2tm takes the a coefficients of an AR model and create
% the appropriate state transfer matrix A
% Usage: A=ar2tm(a) ;

AR_model_order = length(AR_parameters) ;
state_transition_matrix = zeros(AR_model_order) ;
state_transition_matrix(AR_model_order,:) = AR_parameters(AR_model_order:-1:1) ;

for i=1:AR_model_order-1 
    state_transition_matrix(i,i+1)=1;
end 

