%test carrier synchronizer
%Correct phase and frequency offsets of a QPSK signal passed through an AWGN channel.

%Create a phase and frequency offset System object™, where the frequency offset is 1% of the sample rate.
phase_frequency_object_object = comm.PhaseFrequencyOffset(...
                                                    'FrequencyOffset',1e4,...
                                                    'PhaseOffset',45,...
                                                    'SampleRate',1e6);

%Create a carrier synchronizer object:
carrier_synchronizer_object = comm.CarrierSynchronizer( ...
                                        'SamplesPerSymbol',1,...
                                        'Modulation','QPSK');

%Generate random data symbols and apply QPSK modulation.
symbol_stream = randi([0 3],10000,1);
modulated_baseband = pskmod(symbol_stream,4,pi/4);

%Apply phase and frequency offsets using the pfo System object. Then, pass the offset signal through an AWGN channel.
modulated_baseband_offset = step(phase_frequency_object_object, modulated_baseband);
received_baseband = awgn(modulated_baseband_offset,15);

%Display the scatter plot of the received signal. The data appear in a circle instead of being grouped around the reference constellation points due to the frequency offset.
scatterplot(received_baseband)

%Correct for the phase and frequency offset by using the carrier synchronizer object.
received_baseband_carrier_synchronized = step(carrier_synchronizer_object,received_baseband);

%Display the first 1000 symbols of corrected signal. The synchronizer has not yet converged so the data is not grouped around the reference constellation points.
scatterplot(received_baseband_carrier_synchronized(1:1000))

%Display the last 1000 symbols of the corrected signal. The data is now aligned with the reference constellation because the synchronizer has converged to a solution.
scatterplot(received_baseband_carrier_synchronized(9001:10000))






