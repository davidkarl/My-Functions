% %%******Example: Robuts Echo cancellation
% %Algorithm performs standard echo cancelation at t<0 and than
% %a sudden cross-talk inteference is added at t=0.
% 

% x      -measurement data
% d      -desired response or reference signal (same size as x)
% M      -empty []
% beta   -[]
% alpha  -Scalar value of the magnitude of the pole in the filter (0<= alpha <=1).
%         Alpha of 0 is equivalent to the standard classical GAL filter.
% k      -Reflection coefficients of the trained lattice.
% h      -Coefficients of the second stage FIR filter.
% Function returns:
%
% err     -Error between filter's output and reference signal
% y       -Filter's output
% h       -[]
% k       -[]

number_of_samples=2^12;
filter_denominator = [1.0000 -1.8976 1.1870 -0.0642 -0.2718 0.0858];
h2 = [1.0000 -2.0134 1.4571 -0.2602 -0.2377 0.1015];
Fs = 20;
epsi = 10^-2;
filter_order = 10;
t_min = -50;
t_vec = linspace(t_min,t_min+number_of_samples/Fs,number_of_samples);
[trash,center_sample] = min(abs(t_vec));
Signal_estimate = zeros(1,number_of_samples);
 
for i=1:20
    %%%SIGNAL GENERATION
    %No cross-talk Phase
    signal_sawtooth = sawtooth(t_vec,0.5);
    noise_term = 2*randn(1,number_of_samples);
    desired_signal = signal_sawtooth + noise_term;
    noise_term_filtered = filter(1,filter_denominator,noise_term);

    %Add Cross-Talk Component to reference signal
    %the cross talk is made up of the filtered noise term plus filtered signal from it's middle:
    cross_talk_full = noise_term_filtered;
    cross_talk_numerator_size = 10;
    cross_talk_numerator = ones(cross_talk_numerator_size,1)./cross_talk_numerator_size;
    %add filtered signal to cross talk term from the middle:
    talk = filter(cross_talk_numerator , 1 , signal_sawtooth(center_sample:end));
    cross_talk_full(center_sample:end) = cross_talk_full(center_sample:end) + talk * std(cross_talk_full)./std(talk);
    %desired signal cross talk:
    cross_talk_desired_full = desired_signal;

    %%%TECHNIQUE%%%%
    %Cross Talk Resistance - using Two GAL Filters
    cross_talk_first_half = cross_talk_full(1:center_sample-1);
    desired_signal_first_half = cross_talk_desired_full(1:center_sample-1);
    
    %Initial filter- Estimates Noise TF and Freeze
% beta   -Forgetting factor (0<=beta<=1). beta =1 remembers all the data.
% alpha  -Scalar value of the magnitude of the pole in the filter (0<= alpha <=1).
%         Alpha of 0 is equivalent to the standard classical GAL filter.
%epsi    -small positive constant that initializes the algorithm (epsi <<1)
%k0      -optional 1xM+1 initial reflection coefficients vector
%h0      -optional 1xM+2 initial FIR ladder coefficient vector
    smoothing_factor = 1; %remember all data
    pole_magnitude = 0; %classical GAL filter
    epsi = 10^-3;
    k0_initial_reflection_coefficients = [];
    h0_initial_FIR_ladder_coefficient = [];
    
    [error_over_time,prediction_over_time,FIR_ladder_coefficients,lattice_coefficients] = ...
        gall(cross_talk_first_half , desired_signal_first_half , filter_order,...
            smoothing_factor , pole_magnitude , epsi , k0_initial_reflection_coefficients , h0_initial_FIR_ladder_coefficient);
    
    % err     -learning error curve for the algorithm
% y       -prediction curve (dhat)
% h       -FIR coefficients of the ladder
% k       -Lattice coefficients
    
    
    [s_est3,prediction_over_time,trash,trash] = gall(cross_talk_full,cross_talk_desired_full,...
                    [],[],0,[],lattice_coefficients,FIR_ladder_coefficients);
    s_est3 = s_est3(1:end-1,end);


    %Estimate XTalk Interference
    [n_est,xtalk_est,FIR_ladder_coefficients,lattice_coefficients] = gall(...
        s_est3(center_sample:end-1,end),prediction_over_time(center_sample:end),filter_order,0.9975,0.001,10^-3,[],[]);
    s_est = s_est3';

    %s_est4(N0:end)=[zeros(1,w-1) dtalk(N0:end-w+1)]' - n_est4(1:end,end);
    s_est(center_sample:end) = cross_talk_desired_full(center_sample:end)' - n_est(1:end,end);
    s_est(center_sample:end) = [s_est(center_sample+filter_order:end) zeros(1,filter_order)];
    Signal_estimat = s_est./i + (i-1).*Signal_estimat./i;
end

%Plot
figure
plot(t_vec,Signal_estimat,'r');
hold on;
plot(t_vec,signal_sawtooth)
legend('Gall XTR', 'Desired Response')