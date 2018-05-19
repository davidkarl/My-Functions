%test carrier synchronizer2
%Estimate the frequency offset introduced into a noisy 8-PSK signal using the carrier synchronizer System object™.


%Set the example parameters:
modulation_order = 8;                  % Modulation order
Fs = 1e6;               % Sample rate (Hz)
frequency_offset = 1000;         % Frequency offset (Hz)
phase_offset = 15;       % Phase offset (deg)
SNR_dB = 20;             % Signal-to-noise ratio (dB)

%Create a phase frequency offset object to introduce offsets to a modulated signal:
phase_frequency_offset_object = comm.PhaseFrequencyOffset(...
                                                'FrequencyOffset',frequency_offset,...
                                                'PhaseOffset',phase_offset,...
                                                'SampleRate',Fs);

%Create a carrier synchronizer object to correct for the phase and frequency offsets. 
%Set the Modulation property to 8PSK:
carrier_synchronizer_object = comm.CarrierSynchronizer('Modulation','8PSK');

%Generate random data and apply 8-PSK modulation:
symbol_stream = randi([0 modulation_order-1],5000,1);
modulated_baseband = pskmod(symbol_stream,modulation_order,pi/modulation_order);

%Introduce offsets to the signal and add white noise:
received_baseband_offset = step(phase_frequency_offset_object, modulated_baseband);
received_baseband_offset_noisy = awgn(received_baseband_offset,SNR_dB);

%Use the carrier synchronizer to estimate the phase offset of the received signal:
[~,phase_error] = carrier_synchronizer_object(received_baseband_offset_noisy);

%Determine the frequency offset by using the diff function to compute an approximate derivative of the phase error.
%The derivative must be scaled by 2*pi because the phase error is measured in radians.
estimated_frequency_offset = diff(phase_error)*Fs/(2*pi);

%Plot the running mean of the estimated frequency offset. After the synchronizer converges to a solution, 
%the mean value of the estimate is approximately equal to the input value of 1000 Hz:
estimated_frequency_offset_over_time = cumsum(estimated_frequency_offset)./(1:length(estimated_frequency_offset))';
plot(estimated_frequency_offset_over_time)
xlabel('Symbols')
ylabel('Estimated Frequency Offset (Hz)')
grid







