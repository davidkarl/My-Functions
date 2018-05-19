% Covariance analysis of radar tracking problem, 
% given as Example 4.4 in 
% M. S. Grewal and A. P. Andrews, 
% Kalman Filtering: Theory and Practice, 
% John Wiley & Sons, 2000. 
% 
clf; 
disp('Covariance analysis of radar tracking problem,'); 
disp('given as Example 4.4 in'); 
disp('M. S. Grewal and A. P. Andrews,'); 
disp('Kalman Filtering: Theory and Practice,'); 
disp('John Wiley & Sons, 2000.'); 
disp(' '); 
disp('Plots histories of six mean squared state'); 
disp('uncertainties and six magnitudes of Kalman gains'); 
disp('for intersample intervals of 5, 10 and 15 seconds.'); 
disp(' '); 
disp('Six state variables:'); 
disp('  1. Range to object being tracked.'); 
disp('  2. Range rate of object being tracked.'); 
disp('  3. Object range maneuvering noise (pseudo state).'); 
disp('  4. Bearing to object being tracked.'); 
disp('  5. Bearing rate of object being tracked.'); 
disp('  6. Object bearing maneuvering noise (pseudo state).'); 
disp('Pseudo states are used for modeling correlated noise.'); 

%Initialize pseudo state noise variances:
sigma_squared_pseudo_state_WN_variance1 = (103/3)^2; 
sigma_squared_pseudo_state_WN_variance2 = 1.3E-8; 
%Initialize measurement noise variances:
sigma_squared_range_variance = (1000)^2; 
sigma_squared_bearing_variance = (.017)^2; 
state_noise_correlation_coefficient = 0.5; % rho = E[U(k)U(k-1)]/sigma_m^2 = 1-lambda*T, for T<1/lambda

%Initialize State Transition matrix (the part not(!!!) depending on T) : 
Phi = eye(6); 
Phi(2,3) = 1; 
Phi(5,6) = 1; 
Phi(3,3) = state_noise_correlation_coefficient; 
Phi(6,6) = state_noise_correlation_coefficient; 

%Initialize state noise covariance:
Q        = zeros(6);
Q(3,3)   = sigma_squared_pseudo_state_WN_variance1;
Q(6,6)   = sigma_squared_pseudo_state_WN_variance2;
%Initialize measurement noise covariance:
R        = zeros(2);
R(1,1)   = sigma_squared_range_variance;
R(2,2)   = sigma_squared_bearing_variance;
%Initialize measurement (or sensitivity) matrix:
H        = zeros(2,6);
H(1,1)   = 1;
H(2,4)   = 1;


%Initialize time vec:
time_vec = zeros(3,32); % time (for 3 plots)

%Initialize state covariances:
range_covariance = zeros(3,32); % Range covariance
range_derivative_covariance = zeros(3,32); % Range Rate covariance
bearing_covariance = zeros(3,32); % Bearing covariance
bearing_derivative_covariance = zeros(3,32); % Bearing Rate covariance
range_derivative_noise_covariance = zeros(3,32); % Range Rate Noise covariance
bearing_derivative_noise_covariance = zeros(3,32); % Bearing Rate Noise covariance

%Initialize kalman gains:
kalman_gain_range = zeros(3,32); % Range Kalman gain
kalman_gain_range_derivative = zeros(3,32); % Range Rate Kalman gain
kalman_gain_bearing = zeros(3,32); % Bearing Kalman gain
kalman_gain_bearing_derivative = zeros(3,32); % Bearing Rate Kalman gain
kalman_gain_range_derivative_noise = zeros(3,32); % Range Rate Noise Kalman gain
kalman_gain_bearing_derivative_noise = zeros(3,32); % Bearing Rate Noise Kalman gain


time_counter = 0;
%Loop over three possible values of measurement interval / tracking interval:
for T = 5:5:15,
    time_counter = time_counter+1;
    disp(['Simulating tracking at ',num2str(T),' second intervals.']);
    
    %Update state transition matrix according to time T:
    Phi(1,2) = T;
    Phi(4,5) = T;
    
    %Initialize estimate covariance matrix:
    P        = zeros(6);
    P(1,1)   = sigma_squared_range_variance;
    P(1,2)   = sigma_squared_range_variance/T;
    P(2,1)   = P(1,2);
    P(2,2)   = 2*sigma_squared_range_variance/T^2 + sigma_squared_pseudo_state_WN_variance1;
    P(3,3)   = sigma_squared_pseudo_state_WN_variance1;
    P(4,4)   = sigma_squared_bearing_variance;
    P(4,5)   = sigma_squared_bearing_variance/T;
    P(5,4)   = P(4,5);
    P(5,5)   = 2*sigma_squared_bearing_variance/T^2 + sigma_squared_pseudo_state_WN_variance2;
    P(6,6)   = sigma_squared_pseudo_state_WN_variance2;
    
    %Loop over consecutive kalman prediction-update cycles for the current tracking interval:
    for cycle=0:15,

        % Save a priori values:
        prior      = 2*cycle+1;
        time_vec(time_counter,prior) = T*cycle;
        
        %Calculate the Kalman Gain Matrix:
        Kalman_gain_matrix = P*H'/(H*P*H'+R); 
        
        %Track kalman gain over time:
        kalman_gain_range(time_counter,prior) = sqrt(Kalman_gain_matrix(1,1)^2+Kalman_gain_matrix(1,2)^2); % Range Kalman gain
        kalman_gain_range_derivative(time_counter,prior) = sqrt(Kalman_gain_matrix(2,1)^2+Kalman_gain_matrix(2,2)^2); % Range Rate Kalman gain
        kalman_gain_range_derivative_noise(time_counter,prior) = sqrt(Kalman_gain_matrix(3,1)^2+Kalman_gain_matrix(3,2)^2); % Range Rate Noise Kalman gain
        kalman_gain_bearing(time_counter,prior) = sqrt(Kalman_gain_matrix(4,1)^2+Kalman_gain_matrix(4,2)^2); % Bearing Kalman gain
        kalman_gain_bearing_derivative(time_counter,prior) = sqrt(Kalman_gain_matrix(5,1)^2+Kalman_gain_matrix(5,2)^2); % Bearing Rate Kalman gain
        kalman_gain_bearing_derivative_noise(time_counter,prior) = sqrt(Kalman_gain_matrix(6,1)^2+Kalman_gain_matrix(6,2)^2); % Bearing Rate Noise Kalman gain
        
        %Track state estimation covariances over time:
        range_covariance(time_counter,prior) = P(1,1); % Range covariance
        range_derivative_covariance(time_counter,prior) = P(2,2); % Range Rate covariance
        bearing_covariance(time_counter,prior) = P(4,4); % Bearing covariance
        bearing_derivative_covariance(time_counter,prior) = P(5,5); % Bearing Rate covariance
        range_derivative_noise_covariance(time_counter,prior) = P(3,3); % Range Rate Noise covariance
        bearing_derivative_noise_covariance(time_counter,prior) = P(6,6); % Bearing Rate Noise covariance
        
        %Save a posteriori values:
        post  = prior + 1;
        time_vec(time_counter,post) = T*cycle;
        
        %Update state estimation covariance matrix and keep it symmetric:
        P = P - Kalman_gain_matrix*H*P;
        P = 0.5*(P+P');
        
        %Track state estimation covariance over time:
        range_covariance(time_counter,post)   = P(1,1); % Range covariance
        range_derivative_covariance(time_counter,post)  = P(2,2); % Range Rate covariance
        bearing_covariance(time_counter,post)   = P(4,4); % Bearing covariance
        bearing_derivative_covariance(time_counter,post)  = P(5,5); % Bearing Rate covariance
        range_derivative_noise_covariance(time_counter,post) = P(3,3); % Range Rate Noise covariance
        bearing_derivative_noise_covariance(time_counter,post) = P(6,6); % Bearing Rate Noise covariance
        
        %Track Kalman gain over time:
        kalman_gain_range(time_counter,post)    = sqrt(Kalman_gain_matrix(1,1)^2+Kalman_gain_matrix(1,2)^2); % Range Kalman gain
        kalman_gain_range_derivative(time_counter,post)   = sqrt(Kalman_gain_matrix(2,1)^2+Kalman_gain_matrix(2,2)^2); % Range Rate Kalman gain
        kalman_gain_bearing(time_counter,post)    = sqrt(Kalman_gain_matrix(4,1)^2+Kalman_gain_matrix(4,2)^2); % Bearing Kalman gain
        kalman_gain_bearing_derivative(time_counter,post)   = sqrt(Kalman_gain_matrix(5,1)^2+Kalman_gain_matrix(5,2)^2); % Bearing Rate Kalman gain
        kalman_gain_range_derivative_noise(time_counter,post)  = sqrt(Kalman_gain_matrix(3,1)^2+Kalman_gain_matrix(3,2)^2); % Range Rate Noise Kalman gain
        kalman_gain_bearing_derivative_noise(time_counter,post)  = sqrt(Kalman_gain_matrix(6,1)^2+Kalman_gain_matrix(6,2)^2); % Bearing Rate Noise Kalman gain
        
        %Update state estimation covariance matrix:
        P = Phi*P*Phi' + Q;
    end
    
end
figure(1)
subplot(2,2,1),plot(time_vec(1,:),range_covariance(1,:),time_vec(2,:),range_covariance(2,:),time_vec(3,:),range_covariance(3,:));xlabel('Time (sec)');ylabel('Range Cov');
subplot(2,2,2),plot(time_vec(1,:),range_derivative_covariance(1,:),time_vec(2,:),range_derivative_covariance(2,:),time_vec(3,:),range_derivative_covariance(3,:));xlabel('Time (sec)');ylabel('Range Rate Cov'); 
subplot(2,2,3),plot(time_vec(1,:),bearing_covariance(1,:),time_vec(2,:),bearing_covariance(2,:),time_vec(3,:),bearing_covariance(3,:));xlabel('Time (sec)');ylabel('Bear. Cov'); 
subplot(2,2,4),plot(time_vec(1,:),bearing_derivative_covariance(1,:),time_vec(2,:),bearing_derivative_covariance(2,:),time_vec(3,:),bearing_derivative_covariance(3,:));xlabel('Time (sec)');ylabel('Bear. Rate Cov'); 

figure(2)
disp('Figures 4.13-16 on pages 150--151.') 
subplot(2,2,1),plot(time_vec(1,:),range_derivative_noise_covariance(1,:),time_vec(2,:),range_derivative_noise_covariance(2,:),time_vec(3,:),range_derivative_noise_covariance(3,:));xlabel('Time (sec)');ylabel('RRateNoise Cov'); 
subplot(2,2,2),plot(time_vec(1,:),bearing_derivative_noise_covariance(1,:),time_vec(2,:),bearing_derivative_noise_covariance(2,:),time_vec(3,:),bearing_derivative_noise_covariance(3,:));xlabel('Time (sec)');ylabel('BRateNoise Cov'); 
subplot(2,2,3),plot(time_vec(1,:),kalman_gain_range(1,:),time_vec(2,:),kalman_gain_range(2,:),time_vec(3,:),kalman_gain_range(3,:));xlabel('Time (sec)');ylabel('Range Gain'); 
subplot(2,2,4),plot(time_vec(1,:),kalman_gain_range_derivative(1,:),time_vec(2,:),kalman_gain_range_derivative(2,:),time_vec(3,:),kalman_gain_range_derivative(3,:));xlabel('Time (sec)');ylabel('Range Rate GAin'); 

figure(3)
disp('Figures 4.17-20 on pages 152--153.') 
subplot(2,2,1),plot(time_vec(1,:),kalman_gain_bearing(1,:),time_vec(2,:),kalman_gain_bearing(2,:),time_vec(3,:),kalman_gain_bearing(3,:));xlabel('Time (sec)');ylabel('Bear. Gain'); 
subplot(2,2,2),plot(time_vec(1,:),kalman_gain_bearing_derivative(1,:),time_vec(2,:),kalman_gain_bearing_derivative(2,:),time_vec(3,:),kalman_gain_bearing_derivative(3,:));xlabel('Time (sec)');ylabel('Bear Rate Gain'); 
subplot(2,2,3),plot(time_vec(1,:),kalman_gain_range_derivative_noise(1,:),time_vec(2,:),kalman_gain_range_derivative_noise(2,:),time_vec(3,:),kalman_gain_range_derivative_noise(3,:));xlabel('Time (sec)');ylabel('RRateNoiseGain'); 
subplot(2,2,4),plot(time_vec(1,:),kalman_gain_bearing_derivative_noise(1,:),time_vec(2,:),kalman_gain_bearing_derivative_noise(2,:),time_vec(3,:),kalman_gain_bearing_derivative_noise(3,:));xlabel('Time (sec)');ylabel('BRateNoiseGain'); 

