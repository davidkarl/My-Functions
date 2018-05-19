function [ys,e,kout]=kalman_filter_calculate_steady_state_P_then_filter(input_signal,A_state_transition_matrix)
% Performs steady-state Kalman filtering
% assumes z=ys+e, where e is the innovation and ys is generated
% by the x(t+1)=A*x(t), ys=[0..0 1]*x discrete system. 
% First estimate measurement noise, than iteratively estimates 
% the process noise and calls kalmk
% Usage: [ys,e,kout]=kalmke(z,A,k)
% ys,e,z,k,kout are row vectors
% 

T=length(input_signal) ;
n=length(A_state_transition_matrix) ;

% Fix for unstable matrixes
while (max(abs(roots(poly(A_state_transition_matrix))))>0.999)
        A_state_transition_matrix=0.99*A_state_transition_matrix; 
end

number_of_auto_correlation_coefficients = max(50,n) ; % number of autocorr coefs used
precision_tolerance = 1e-2 ;   % the desired precision
r = 1;

   opt = foptions; 
   opt(1) = 0; 
   opt(2) = precision_tolerance;
   
   i = fmin('kbfopt',-20,log(r)/log(10),opt,A_state_transition_matrix,r,input_signal) ;  
   q=10^(i) ;
   P=eye(n) ;
   Pold=2*P ;
   while max(max(abs(P-Pold)))>1e-6,
      Pold=P ;	
%     P=A*(P-P*h'*inv(h*P*h'+r)*h*P)*A'+h*q*h' ;
      P=A_state_transition_matrix*(P-P(:,n)/(P(n,n)+r)*P(n,:))*A_state_transition_matrix' ;
      P(n,n)=P(n,n)+q ;
      end ;
%   P
   k=P(:,n)/(P(n,n)+r) ;
   [ys e]=kalmk(input_signal,A_state_transition_matrix,k') ;   
   kout=k' ;





