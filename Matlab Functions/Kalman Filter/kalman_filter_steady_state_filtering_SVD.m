function [y_signal_estimate,residual_error] = ...
        kalman_filter_steady_state_filtering_SVD(input_signal,state_transition_matrix,kalman_gain)
% Performs steady-state Kalman filtering
% assumes z=ys+e, where e is the innovation and ys is generated
% by the x(t+1)=Ax(t), ys=[0..0 1] x discrete system. 
% k is the Kalman gain
% Usage: [ys,e]=kalmk(z,A,k)
% ys,e,z,k are row vectors

input_signal_length = length(input_signal);
state_vector_length = length(state_transition_matrix);
kalman_gain = kalman_gain';

M = zeros(input_signal_length,state_vector_length) ; % M*x0-p = err
p = zeros(input_signal_length,1) ;   % the right side of the equation
yhat = zeros(input_signal_length) ;
w = eye(state_vector_length);    % the actual estimate
wa = zeros(state_vector_length,1);         % the absolute factor     
IK = zeros(state_vector_length); 
IK(:,state_vector_length) = kalman_gain; 
IK = state_transition_matrix*(eye(state_vector_length)-IK);

for i=1:input_signal_length,
    M(i,:) = -w(state_vector_length,:); 
    p(i) = wa(state_vector_length) ;
    % Compute y(k|k-1) -> y(k+1|k)
    wa = state_transition_matrix*( wa + kalman_gain*(input_signal(i)-wa(state_vector_length)) ) ;
    w = IK*w ;
end
p = input_signal' - p ;

% x0=M \ p ;
% To get rid of the warnings

[U,S,V] = svd(M,0);

%WHAT'S GOING ON HERE?????
ms = max(diag(S))*1e-12 ;
for i=1:state_vector_length, 
    if S(i,i) > ms
        S(i,i) = 1/S(i,i);
    else
        S(i,i)=0;
    end
end

x0 = V*S*U'*p;

residual_error = p - M*x0; 
y_signal_estimate = input_signal - residual_error';


 
 





