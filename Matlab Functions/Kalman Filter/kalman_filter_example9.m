% Nonlinear estimation of damping coefficient. 
% 
% Example 5.3 in 
% Kalman Filtering: Theory and Practice, 
% by M. S. Grewal and A. P. Andrews, 
% published by Wiley, 2000. 
% 
% This implementation adds dynamic process noise 
% (plant noise) to improve observability of the 
% damping factor. 
% 
clear all; 
close all; 
disp('See Example 5.3 in'); 
disp('Kalman Filtering: Theory and Practice,'); 
disp('by M. S. Grewal and A. P. Andrews,'); 
disp('published by Wiley, 2000.'); 
disp(' '); 
disp('Demonstration of extended Kalman filter'); 
disp('estimating the position, velocity and'); 
disp('damping factor of a damped harmonic'); 
disp('oscillator with constant forcing.'); 
disp(' '); 
disp('This would be an appropriate model for'); 
disp('a VERTICAL mass-spring system with UNKNOWN'); 
disp('constant damping and known gravity forcing.'); 
disp(' '); 
disp('This particular implementation uses dynamic'); 
disp('disturbance noise (plant noise) to improve'); 
disp('observability of the damping coefficient'); 
disp(' '); 
% 
% Model parameters 
% 
zeta  = 0.2;     % TRUE damping factor (unitless) 
I3    = eye(3);  % (identity matrix) 
T     = .01;     % intersample time interval (s) 
                 % (not specified in Example 5.3) 
qc    = 4.47;    % continuous process noise covariance (ft^2/s^3) 
R     = .01;     % measurement noise variance (ft^2) 
omega = 5;       % undamped resonant frequency (rad/s) 
f     = [0;12;0];% non-homogeneous forcing (constant) 
H     = [1,0,0]; % measurement sensitivity matrix 
lambda = exp(-T*omega*zeta); 
psi    = 1 - zeta^2; 
xi     = sqrt(psi); 
theta  = xi*omega*T; 
c      = cos(theta); 
s      = sin(theta); 
Phix    = zeros(3); 
% 
% This implementation uses the exact solution for 
% them matrix exponential, exp(T*F) for the state 
% transition matrix, derived using  
% symbolic math software.  The values can also be 
% calculated numerically in Matlab using Runge-Kutta 
% integration. 
% 
Phix(1,1) = lambda*c + zeta*s/xi; 
Phix(1,2) = lambda*s/(omega*xi); 
Phix(2,1) = -omega*lambda*s/xi; 
Phix(2,2) = lambda*c - zeta*s/xi; 
Phix(3,3) = 1; 
% 
% First-order approximation 
% 
Phix   = I3 + T*[0,1,0;-omega^2,-2*omega*zeta,0;0,0,0]; 
% 
% The discrete time forcing term is 
% 
delta = f(2)*[(1 - lambda*(c-zeta*s*xi/psi))/omega^2; 
              lambda*s/(omega*xi); 
              0]; 
% 
% The EKF approximation of Qd will be T*Qc 
% 
Qd     = T*[0,0,0;0,qc,0;0,0,0]; 
sigma  = sqrt(Qd(2,2)); 
disp('(Allow a moment for simulation.)'); 
% 
x     = [0;0;zeta]; 
P     = [2,0,0;0,2,0;0,0,.1]; 
H     = [1,0,0]; 
R     = .01; 
xh    = [0;0;0]; 
m     = 0; 
%t=0;x1=0;x2=0;x3=0;xh1=0;xh2=0;xh3=0;p11=0;p22=0;p33=0; 
   for k=1:101, 
   t(k)   = T*(k-1); 
   x1(k)  = x(1); 
   x2(k)  = x(2); 
   x3(k)  = x(3); 
% 
% a priori values 
% 
   m      = m+1;  
   tm(m)  = t(k); 
   xh1(m) = xh(1); 
   xh2(m) = xh(2); 
   xh3(m) = xh(3); 
   p11(m) = P(1,1); 
   p22(m) = P(2,2); 
   p33(m) = P(3,3); 
% 
% a posteriori values following observational update 
% 
   m      = m+1;  
   z      = x(1); 
   K      = P*H'/(H*P*H'+R); 
   xh     = xh + K*(z-H*xh); 
   P      = P - K*H*P; 
   P      = .5*(P+P'); 
   tm(m)  = t(k); 
   xh1(m) = xh(1); 
   xh2(m) = xh(2); 
   xh3(m) = xh(3); 
   p11(m) = P(1,1); 
   p22(m) = P(2,2); 
   p33(m) = P(3,3); 
% 
% Temporal update 
% 
   x      = Phix*x + delta + sigma*[0;randn;0]; % update of true state with noise    
% 
% The EKF approximation for Phi is I3 + T*F, 
% where, for (d/dt) x(t) = a(x(t)), F is the 
% partial derivative F = (d/dx)a(x) evaluated 
% at the estimated value of x. 
% 
   Phih   = I3 + T*[0,1,0; 
                    -omega^2,-2*omega*xh(3),-2*omega*xh(2); 
                    0,0,0]; 
   xh    = Phih*xh + delta; 
   P      = Phih*P*Phih' + Qd; 
   end; 
clf; 
subplot(3,1,1),plot(t,x1,'b-',tm,xh1,'g-'); 
%legend('True','Est.'); 
xlabel('Time (sec)');ylabel('Position (ft)');title('True and estimated states. (Press <ENTER> to continue.)'); 
subplot(3,1,2),plot(t,x2,'b-',tm,xh2,'g-'); 
%legend('True','Est.'); 
xlabel('Time (sec)');ylabel('Velocity (fps)'); 
subplot(3,1,3),plot(t,x3,'b-',tm,xh3,'g-'); 
%legend('True','Est.'); 
xlabel('Time (sec)');ylabel('Damp. Factor'); 
disp('Press <ENTER> to continue.'); 
pause 
subplot(3,1,1),plot(tm,p11);xlabel('Time (sec)');ylabel('Position');title('Mean squared estimation uncertainties'); 
subplot(3,1,2),plot(tm,p22);xlabel('Time (sec)');ylabel('Velocity'); 
subplot(3,1,3),plot(tm,p33);xlabel('Time (sec)');ylabel('Damp. Factor'); 
disp('DONE');