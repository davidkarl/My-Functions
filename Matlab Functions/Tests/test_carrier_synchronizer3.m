%test carrier synchronizer 3
%Correct phase and frequency offsets for a QAM signal in an AWGN channel. Coarse frequency estimator and carrier synchronizer System objects™ are used to compensate for a significant offset.


%Set the example parameters:
Fs = 10000;           % Sample rate (Hz)
samples_per_symbol = 4;              % Samples per symbol
modulation_order = 16;               % Modulation order
bits_per_symbol = log2(modulation_order);          % Bits per symbol

%Create an AWGN channel System object™ (i give it bits_per_symbol because it's SNR is stated for Eb/No):
AWGN_object = comm.AWGNChannel('EbNo',15,'BitsPerSymbol',bits_per_symbol,'SamplesPerSymbol',samples_per_symbol);

%Create a pulse shaping filter:
raised_cosine_transmitter_object = comm.RaisedCosineTransmitFilter(...
                                         'OutputSamplesPerSymbol',samples_per_symbol);
raised_cosine_receiver_object = comm.RaisedCosineReceiveFilter(...
                                         'InputSamplesPerSymbol',samples_per_symbol, ...
                                         'DecimationFactor', samples_per_symbol);

%Create a constellation diagram object to visualize the effects of the carrier synchronization:
constellation_diagram_object = comm.ConstellationDiagram(...
    'ReferenceConstellation',qammod(0:modulation_order-1,modulation_order), ...
    'XLimits',[-5 5],'YLimits',[-5 5]);

%Create a QAM coarse frequency estimator to roughly estimate the frequency offset. 
%This is used to reduce the frequency offset of the signal passed to the carrier synchronizer. 
%In this case, a frequency estimate to within 10 Hz is sufficient.
QAM_coarse_frequency_estimator_object = comm.QAMCoarseFrequencyEstimator('SampleRate',Fs, ...
                                                                         'FrequencyResolution',10);

%Create a carrier synchronizer System object. 
%Because of the coarse frequency correction, the carrier synchronizer will converge quickly even 
%though the normalized bandwidth is set to a low value. 
%Lower normalized bandwidth values enable better correction.
carrier_synchronizer_object = comm.CarrierSynchronizer( ...
                                                'DampingFactor',0.7, ...
                                                'NormalizedLoopBandwidth',0.005, ...
                                                'SamplesPerSymbol',samples_per_symbol,...
                                                'Modulation','QAM');

%Create phase and frequency offset objects. pfo is used to introduce a phase and frequency 
%offset of 30 degrees and 250 Hz, respectively. pfc is used to correct the offset in the received signal by using the output of the coarse frequency estimator.
phase_Frequency_offset_object = comm.PhaseFrequencyOffset(...
                                                'FrequencyOffset',250,...
                                                'PhaseOffset',30,...
                                                'SampleRate',Fs);

phase_frequency_compensator_object = comm.PhaseFrequencyOffset('FrequencyOffsetSource','Input port', ...
                                                'SampleRate',Fs);

%Generate random data symbols and apply 16-QAM modulation:
symbol_stream = randi([0 modulation_order-1],10000,1);
modulated_baseband = qammod(symbol_stream,modulation_order);
modulated_baseband_filtered = step(raised_cosine_transmitter_object, modulated_baseband);

%Pass the signal through an AWGN channel and apply a phase and frequency offset:
modulated_baseband_filtered_offset = step(phase_Frequency_offset_object, modulated_baseband_filtered);
received_baseband_noisy = step(AWGN_object, modulated_baseband_filtered_offset);

%Estimate the frequency offset and compensate for it using PFC. 
%Plot the constellation diagram of the output, syncCoarse. 
%From the spiral nature of the diagram, you can see that the phase and frequency offsets are not corrected:
coarse_frequency_offset_estimation = step(QAM_coarse_frequency_estimator_object, received_baseband_noisy);
received_baseband_coarse_frequency_offset_compensated = ...
            step(phase_frequency_compensator_object, received_baseband_noisy, -coarse_frequency_offset_estimation);

%Plot constellation:
step(constellation_diagram_object, received_baseband_coarse_frequency_offset_compensated)

%Apply FINE frequency correction to the signal by using the carrier synchronizer object.
received_baseband_fine_carrier_synchronized = ...
                        step(carrier_synchronizer_object, received_baseband_coarse_frequency_offset_compensated);
received_baseband_fine_carrier_synchronized_filtered = ...
                        step(raised_cosine_receiver_object, received_baseband_fine_carrier_synchronized);

%Display the constellation diagram of the last 1000 symbols. You can see that these symbols are aligned with the reference constellation because the carrier synchronizer has converged to a solution.
release(constellation_diagram_object)
step(constellation_diagram_object, received_baseband_fine_carrier_synchronized_filtered(9001:10000))




