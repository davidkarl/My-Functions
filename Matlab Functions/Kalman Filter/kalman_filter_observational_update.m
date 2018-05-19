function [x,P] = kalman_filter_observational_update(x,P,z,H,R)
% 
% function [x,P] = obsup[x,P,z,H,R] 
% 
% Performs Kalman filter observational update of state vector, x 
% and the associated covriance matrix of estimation uncertainty, P 
% 
% Inputs  x is n-by-1 (column) state vector 
%         P is n-by-n symmetric pos. def. matrix 
%         z is m-by-1 measurement (observation) vector 
%         H is m-by-n measurement sensitivity matrix 
%         R is m-by-m covariance matrix of meas. uncertainty 
% Outputs x,P updated by observation y 
% 
PHt = P*H'; 
K   = PHt/(H*PHt+R); 
x   = x + K*(z-H*x); 
P   = P - K*PHt'; 