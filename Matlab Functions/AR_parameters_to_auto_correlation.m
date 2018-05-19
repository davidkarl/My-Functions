function [auto_correlation_sequence] = AR_parameters_to_auto_correlation(AR_parameters,auto_correlation_length)
% Given AR model coefficients a, calculate first m autocorrelation
% coefficients (including zero lag one stored in c(1))
% Assume white input noise with unit dispersion (!!!!) 
% Usage c=a2c(a,m) ;

n = length(AR_parameters) ; 
state_transition_matrix = AR_parameters_to_state_transfer_matrix(AR_parameters) ;

P = ones(n) ;
h = [ zeros(1,n-1) , 1 ]; 
G = h'*h ;

%Solve discrete Lyapunov equation (A*X*A' - X + G = 0)  (GO OVER THIS AGAIN!!!!!):
P = dlyap1(state_transition_matrix,G); 
    
%Initialize Ak and auto_correlation_sequence:
Ak = eye(1);
auto_correlation_sequence = zeros(1,auto_correlation_length);

%Loop over auto correlation to auto correlation from AR parameters:
for k=1:auto_correlation_length,
    auto_correlation_sequence(k) = h*Ak*P*h' ;
    Ak=Ak*state_transition_matrix ;
end 
auto_correlation_sequence = auto_correlation_sequence';

















