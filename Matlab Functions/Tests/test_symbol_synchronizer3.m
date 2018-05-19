%test symbol synchronizer 3

%Correct for a monotonically increasing symbol timing error on a noisy 8-PSK signal 
%and display the normalized timing error.

%Set example parameters:
symbol_rate = 5000;                   % Symbol rate (Hz)
samples_per_symbol = 2;                       % Samples per symbol
Fs = samples_per_symbol*symbol_rate;              % Sample rate (Hz)

%Create raised cosine transmit and receive filter System objects™.
raised_cosine_transmitter_object = comm.RaisedCosineTransmitFilter( ...
                                                'OutputSamplesPerSymbol',samples_per_symbol);
raised_cosine_receiver_object = comm.RaisedCosineReceiveFilter( ...
                                                'InputSamplesPerSymbol',samples_per_symbol, 'DecimationFactor',1);

%Create a variable fractional delay object to introduce a monotonically increasing timing error.
variable_fractional_delay_object = dsp.VariableFractionalDelay;

%Create a SymbolSynchronizer object to eliminate the timing error:
symbol_synchronizer_object = comm.SymbolSynchronizer(...
                                     'TimingErrorDetector','Mueller-Muller (decision-directed)', ...
                                     'SamplesPerSymbol',samples_per_symbol);

%Generate random 8-ary symbols and apply PSK modulation:
symbol_stream = randi([0 7],5000,1);
modulated_baseband = pskmod(symbol_stream,8,pi/8);

%Filter the modulated signal through a raised cosine transmit filter and apply a monotonically increasing timing delay.
variable_fractional_delay_vec = (0:1/Fs:1-1/Fs)';
transmitted_baseband = raised_cosine_transmitter_object(modulated_baseband);
transmitted_baseband_delayed = variable_fractional_delay_object(transmitted_baseband,variable_fractional_delay_vec);

%Pass the delayed signal through an AWGN channel with a 15 dB signal-to-noise ratio.
received_baseband = awgn(transmitted_baseband_delayed,15,'measured');

%Filter the received signal and display its scatter plot. The scatter plot shows that the received signal does not align with the expected 8-PSK reference constellation due to the timing error.
received_baseband_noisy = raised_cosine_receiver_object(received_baseband);
scatterplot(received_baseband_noisy,samples_per_symbol);

%Correct for the symbol timing error by using the symbol synchronizer. Display the scatter plot and observe that the synchronized signal now aligns with the 8-PSK constellation.
[received_baseband_symbol_synchronized,timing_error_vec] = symbol_synchronizer_object(received_baseband_noisy);
scatterplot(received_baseband_symbol_synchronized(1001:end))

%Plot the timing error estimate. You can see that the normalized timing error ramps up to 1 sample.
figure
plot(variable_fractional_delay_vec,timing_error_vec)
xlabel('Time (s)')
ylabel('Timing Error (samples)')











