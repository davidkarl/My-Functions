%test symbol synchronizer 2

%Create a QAM modulator and an AWGN channel object.
rectangular_QAM_modulator_object = comm.RectangularQAMModulator('NormalizationMethod','Average power');
AWGN_object = comm.AWGNChannel('NoiseMethod','Signal to noise ratio (SNR)', 'SNR',20);

%Create a matched pair of raised cosine filter objects.
raised_cosine_transmitter_object = comm.RaisedCosineTransmitFilter('FilterSpanInSymbols',10, 'OutputSamplesPerSymbol',8);
raised_cosine_receiver_object = comm.RaisedCosineReceiveFilter('FilterSpanInSymbols',10, 'InputSamplesPerSymbol',8,'DecimationFactor',4);

%Create a PhaseFrequencyOffset object to introduce a 100 Hz Doppler shift.
phase_frequency_offset_object = comm.PhaseFrequencyOffset('FrequencyOffset',100, 'PhaseOffset',45,'SampleRate',1e6);

%Create a variable delay object to introduce timing offsets.
variable_fractional_delay_object = dsp.VariableFractionalDelay;

%Create carrier and symbol synchronizer objects to correct for a Doppler shift and a timing offset, respectively.
carrier_synchronizer_object = comm.CarrierSynchronizer('SamplesPerSymbol',2);
symbol_synchronizer_object = comm.SymbolSynchronizer('TimingErrorDetector','Early-Late (non-data-aided)', ...
                                                    'SamplesPerSymbol',2);

%Create constellation diagram objects to view results:
constellation_diagram_object1 = comm.ConstellationDiagram(...
                                       'ReferenceConstellation',constellation(rectangular_QAM_modulator_object), ...
                                       'SamplesPerSymbol',8,...
                                       'Title','Received Signal');
constellation_diagram_object2 = comm.ConstellationDiagram(...
                                       'ReferenceConstellation',constellation(rectangular_QAM_modulator_object), ...
                                       'SamplesPerSymbol',2,...
                                       'Title','Frequency Corrected Signal');
constellation_diagram_object3 = comm.ConstellationDiagram(...
                                       'ReferenceConstellation',constellation(rectangular_QAM_modulator_object), ...
                                       'SamplesPerSymbol',2,...
                                       'Title','Frequency and Timing Synchronized Signal');
                                   
%Main Processing Loop:
%Perform the following operations:
%(1).Generate random symbols and apply QAM modulation.
%(2).Filter the modulated signal.
%(3).Apply frequency and timing offsets.
%(4).Pass the transmitted signal through an AWGN channel.
%(5).Correct for the Doppler shift.
%(6).Filter the received signal.
%(7). Correct for the timing offset.
for k = 1:15
    %Generate Symbol Stream:
    current_symbol_stream = randi([0 15],2000,1);
    %QAM modulate:
    modulated_baseband = step(rectangular_QAM_modulator_object,current_symbol_stream);                  
    %Transmit filter:
    transmitted_baseband = step(raised_cosine_transmitter_object,modulated_baseband);              
    %Apply Doppler shift:
    transmitted_baseband_offset = step(phase_frequency_offset_object,transmitted_baseband);          
    %Apply variable delay:
    transmitted_baseband_offset_and_delay = step(variable_fractional_delay_object,transmitted_baseband_offset,k/15);    
    
    %Add white Gaussian noise:
    received_baseband = step(AWGN_object,transmitted_baseband_offset_and_delay);                 
    
    %Receive filter:
    received_baseband_filtered = step(raised_cosine_receiver_object,received_baseband);           
    %Correct for Doppler:
    received_baseband_carrier_synchronized = step(carrier_synchronizer_object,received_baseband_filtered);         
    %Correct for timing error:
    received_baseband_carrier_and_symbol_synchronized = ...
                                step(symbol_synchronizer_object,received_baseband_carrier_synchronized);           
end

%Visualization:
%Plot the constellation diagrams of the received signal, the frequency corrected signal, 
%and the frequency and timing synchronized signal. 
%While specific constellation points cannot be indentified in the received signal and 
%only partially identified in the frquency corrected signal, the timing and frequency synchronized 
%signal aligns with the expected QAM constellation points.
step(constellation_diagram_object1,received_baseband)
step(constellation_diagram_object2,received_baseband_carrier_synchronized)
step(constellation_diagram_object3,received_baseband_carrier_and_symbol_synchronized)





