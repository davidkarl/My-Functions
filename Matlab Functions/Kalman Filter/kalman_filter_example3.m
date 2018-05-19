% 
% Example 4.3 in 
% Kalman Filtering: Theory and Practice, 
% by M. S. Grewal and A. P. Andrews, 
% published by John Wiley & Sons, 2000. 
% 
clf; 
disp('Grewal & Andrews,'); 
disp('Kalman Filtering: Theory and Practice'); 
disp('published by John Wiley & Sons, 2000.'); 
disp(' '); 
disp('Example 4.3 demonstration of a Kalman filter'); 
disp('estimating the state (position and velocity) of a damped'); 
disp('harmonic oscillator with constant forcing.'); 
disp(' '); 
disp('This would be an appropriate model for a VERTICAL'); 
disp('mass-spring system with damping and gravity forcing.'); 
disp(' '); 

% The parameters of the problem are:  
T_sample_interval = 0.01; % intersample time interval (s)  
qc_continuous_process_noise_covariance = 4.47; % continuous process noise covariance (ft^2/s^3) 
R_measurement_noise_variance = 0.01; % measurement noise variance (s*ft^2) 
zeta_dampling_factor = 0.2; % damping factor (unitless) 
omega0_undamped_resonant_frequency = 5; % undamped resonant frequency (rad/s) 
f_non_homogenous_force_term = [0;12]; % non-homogeneous forcing (constant) 
H_measurement_matrix = [1,0]; % measurement sensitivity matrix 

% The dynamic coefficient matrix is a function of these: 
% F      = [0,1;-omega^2,-2*zeta*omega]; 
% The exponential e^(T*F) depends on the intermediate values: 
lambda = exp(-T_sample_interval*omega0_undamped_resonant_frequency*zeta_dampling_factor); 
psi    = 1 - zeta_dampling_factor^2; 
xi     = sqrt(psi); 
theta  = xi*omega0_undamped_resonant_frequency*T_sample_interval; 
c      = cos(theta); 
s      = sin(theta); 
Phi    = zeros(2); 

% The exact solution for exp(T*F) is 
Phi(1,1) = lambda*c + zeta_dampling_factor*s/xi; 
Phi(1,2) = lambda*s/(omega0_undamped_resonant_frequency*xi); 
Phi(2,1) = -omega0_undamped_resonant_frequency*lambda*s/xi; 
Phi(2,2) = lambda*c - zeta_dampling_factor*s/xi; 

% and the exact forcing term is (check this) 
delta = f_non_homogenous_force_term(2)*[(1 - lambda*(c-zeta_dampling_factor*s*xi/psi))/omega0_undamped_resonant_frequency^2; 
              lambda*s/(omega0_undamped_resonant_frequency*xi)]; 

% discrete time process noise covariance 
l1  = zeta_dampling_factor^2; 
l4  = exp(-2*omega0_undamped_resonant_frequency*zeta_dampling_factor*T_sample_interval); 
l6  = sqrt(1-l1); 
l8  = l6*omega0_undamped_resonant_frequency*T_sample_interval; 
l9  = cos(2*l8); 
l11 = l4*l9*l1; 
l15 = l4*sin(2*l8)*l6*zeta_dampling_factor; 
l19 = 1/(l1-1); 
l20 = omega0_undamped_resonant_frequency^2; 
l24 = 1/zeta_dampling_factor; 
l32 = l4*(l9-1)*l19/l20; 
Qd  = (qc_continuous_process_noise_covariance/4)*[(l1-l11+l15-1+l4)*l19*l24,l32; 
             l32,(l1-l11-l15-1+l4)*l19*l24/omega0_undamped_resonant_frequency]; 
disp('(Allow a moment for simulation.)'); 

% Initial values: 
xh   = [0;0];       % initial estimate of x 
P    = [2,0;0,2];   % initial covariance of estimation uncertainty 
x    = [0;0];       % initial state; 

% Initialize arrays for simulated values:
m=0;tm=0;t=0;x1=0;x2=0;xh1=0;xh2=0;K1=0;K2=0; 
z=0;sd1=0;sd2=0;cc=0; 

% Simulate and save values (identified by comments) :
for k=1:101,
    t(k)  = (k-1)*T_sample_interval; % Time starting with t=0
    x1(k) = x(1);    % True state variables
    x2(k) = x(2);
    K     = P*H_measurement_matrix'/(H_measurement_matrix*P*H_measurement_matrix'+R_measurement_noise_variance);
    K1(k) = K(1); % Kalman gains
    K2(k) = K(2);
    zn    = x(1)    ; % Measurement (no noise)
    z(k)  = zn;
    %
    % Use double indexing for priori/posteriori values
    %
    m     = m+1;
    tm(m) = t(k);  % time of a priori value
    xh1(m)= xh(1); % a priori estimates
    xh2(m)= xh(2);
    sd1(m)= sqrt(P(1,1));         % Standard deviations
    sd2(m)= sqrt(P(2,2));         % of estimation uncertainty.
    cc(m) = P(1,2)/sd1(m)/sd2(m); % Correlation coeff.
    m     = m+1;
    xh    = xh + K*(zn - H_measurement_matrix*xh); % a posteriori estimate
    P     = P - K*H_measurement_matrix*P;
    tm(m) = t(k);  % Time of a posteriori values.
    xh1(m)= xh(1); % a posteriori estimates
    xh2(m)= xh(2);
    sd1(m)= sqrt(P(1,1));         % Standard deviations
    sd2(m)= sqrt(P(2,2));         % of estimation uncertainty.
    cc(m) = P(1,2)/sd1(m)/sd2(m); % Correlation coeff.
    x     = Phi*x + delta;
    xh    = Phi*xh + delta;
    P     = Phi*P*Phi' + Qd;
    P     = .5*(P+P'); % preserve symmetry of P
end;
disp('Done.');
disp(' '); 
disp('Behold the plots in the accompanying figure.'); 
disp('(If you are running from a windowed environment, you'); 
disp(' may need to click on the figure to make it visible.)'); 
disp('(You may also want to move the figure so that'); 
disp(' it does not cover up this text.)'); 
subplot(2,1,1),plot(t,x1,tm,xh1); 
%legend('True','Est.'); 
xlabel('Time (sec)');ylabel('Position');title('True and estimated states without noise'); 
subplot(2,1,2),plot(t,x2,tm,xh2); 
%legend('True','Est.'); 
xlabel('Time (sec)');ylabel('Velocity'); 
disp(' '); 
disp('These plots show the "true" state dynamics'); 
disp('(without process noise) and estimated values,'); 
disp('which should be nearly equal.'); 
disp(' '); 
disp('Press <ENTER> key to continue'); 
pause 
% 
% The second set of plots are og the rms estimation uncertainties 
% and correlation coefficient as a function of time for the 
% "dynamic noise-free" case (i.e., Q=0 but R = .01). 
% 
subplot(3,1,1),plot(tm,sd1,tm,sd2); 
%legend('Pos','Vel'); 
xlabel('Time (sec)');ylabel('RMS Uncert.');title('Uncertainties and Kalman Gains'); 
subplot(3,1,2),plot(tm,cc);xlabel('Time (sec)');ylabel('Corr. coeff.'); 
subplot(3,1,3),plot(t,K1,t,K2); 
%legend('Pos','Vel'); 
xlabel('Time (sec)');ylabel('Kal. Gains'); 
disp(' '); 
disp('Displayed plots show RMS state uncertainties, correlation'); 
disp('coefficient and Kalman gains, with "sawtooth" curve'); 
disp('shapes showing A PRIORI and A POSTERIORI values.'); 
disp(' '); 
disp('DONE');