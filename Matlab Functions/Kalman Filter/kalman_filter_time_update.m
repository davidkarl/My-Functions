function [x,P] = kalman_filter_time_update(x,P,Phi,Q) 
% 
% function [x,P] = timeup(x,P,Phi,Q) 
% 
% Performs the Kalman filter time update of state vector, x 
% and the associated covariance matrix of estimation uncertainty, P 
% 
% Input  x   is n-dimensional column vector 
%        P   is n-by-n symmetric pos. def. matrix 
%        Phi is n-by-n state transition matrix 
%        Q   is n-by-n covariance matrix of additive state noise 
% Output x and P updated for time passing 
% 
x = Phi*x; 
P = Phi*P*Phi' + Q; 