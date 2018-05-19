%test symbol synchronizer
% Correct for a fixed symbol timing error on a noisy QPSK signal.

%Create raised cosine transmit and receive filter System objects™.
raised_cosine_transmission_object = comm.RaisedCosineTransmitFilter( ...
    'OutputSamplesPerSymbol',4);
raised_cosine_receiver_object = comm.RaisedCosineReceiveFilter( ...
    'InputSamplesPerSymbol',4, ...
    'DecimationFactor',2);

%Create a delay object to introduce a fixed timing error of 2 samples, which is equivalent to 1/2 symbols.
integer_delay_object = dsp.Delay(2);

%Create a SymbolSynchronizer object to eliminate the timing error.
symbol_synchronizer_object = comm.SymbolSynchronizer('TimingErrorDetector','Zero-Crossing (decision-directed)');
                                                                           %'Gardner (non-data-aided)'
                                                                           %'Early-Late (non-data-aided)'
                                                                           %'Mueller-Muller (decision-directed)' 

%Generate random 4-ary symbols and apply QPSK modulation.
symbol_stream = randi([0 3],5000,1);
modulated_signal = pskmod(symbol_stream,4,pi/4);

%Filter the modulated signal through a raised cosine transmit filter and apply a fixed delay.
transmitted_signal = step(raised_cosine_transmission_object, modulated_signal);
transmitted_signal_delayed = step(integer_delay_object, transmitted_signal);

%Pass the delayed signal through an AWGN channel with a 15 dB signal-to-noise ratio.
received_signal = awgn(transmitted_signal_delayed,15,'measured');

%Filter the received signal and display its scatter plot. The scatter plot shows that the received signal does not align with the expected QPSK reference constellation due to the timing error.
received_signal_shaped = step(raised_cosine_receiver_object, received_signal);
scatterplot(received_signal_shaped(1001:end),2)

%Correct for the symbol timing error by using the symbol synchronizer object. Display the scatter plot and observe that the synchronized signal now aligns with the expected QPSK constellation.
received_signal_synchronized = step(symbol_synchronizer_object, received_signal_shaped);
scatterplot(received_signal_synchronized(1001:end),2)









