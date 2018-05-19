%test general QAM communication

% Setup a three point constellation
constellation_points = [0,1];
general_QAM_modulator_object = comm.GeneralQAMModulator(constellation_points);
general_QAM_demodulator_object = comm.GeneralQAMDemodulator(constellation_points);
AWGN_object = comm.AWGNChannel('NoiseMethod', 'Signal to noise ratio (SNR)', 'SNR', 15, 'SignalPower', 0.89);


%Create an error rate calculator
error_rate_object = comm.ErrorRate; 
for counter = 1:100
    % Transmit a 50-symbol frame
    current_symbol_stream = randi([0 1],50,1);
    modulated_baseband = step(general_QAM_modulator_object, current_symbol_stream);
    modulated_baseband_noisy = step(AWGN_object, modulated_baseband);
    demodulated_baseband = step(general_QAM_demodulator_object, modulated_baseband_noisy);
    error_stats = step(error_rate_object, current_symbol_stream, demodulated_baseband);
end
fprintf('Error rate = %f\nNumber of errors = %d\n', ...
    error_stats(1), error_stats(2))



