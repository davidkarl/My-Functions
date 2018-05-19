%test QAM coarse frequency estimator
%Estimate and correct for a -250 Hz frequency offset in a 16-QAM signal using the 
%QAM Coarse Frequency Estimator System object™.


%Create a rectangular QAM modulator System object using name-value pairs to set 
%the modulation order to 16 and the constellation to have an average power of 1 W.
rectangular_QAM_modulator_object = comm.RectangularQAMModulator('ModulationOrder',16, ...
                                                                'NormalizationMethod','Average power', ...
                                                                'AveragePower',1);

%Create a square root raised cosine transmit filter System object.
raised_cosine_transmitter_filter = comm.RaisedCosineTransmitFilter;

%Create a phase frequency offset object, where the FrequencyOffset property is set 
%to -250 Hz and SampleRate is set to 4000 Hz using name-value pairs.
phase_frequency_offset_object = comm.PhaseFrequencyOffset(...
                                                    'FrequencyOffset',-250, ...
                                                    'SampleRate',4000);

%Create a QAM coarse frequency estimator System object with a sample rate of 4 kHz 
%and a frequency resolution of 1 Hz.
QAM_coarse_frequency_estimator_object = comm.QAMCoarseFrequencyEstimator(...
                                        'SampleRate',4000, ...
                                        'FrequencyResolution',1);

%Create a second phase frequency offset object to correct the offset. Set the FrequencyOffsetSource property 
%to Input port so that the frequency correction estimate is an input argument.
phase_frequency_compensator_object = comm.PhaseFrequencyOffset(...
                                        'FrequencyOffsetSource','Input port', ...
                                        'SampleRate',4000);

%Create a spectrum analyzer object to view the frequency response of the signals.
spectrum_analyzer_object = dsp.SpectrumAnalyzer('SampleRate',4000);

%Generate a 16-QAM signal, filter the signal, apply the frequency offset, and pass the signal through the AWGN channel.
modulatd_baseband = step(rectangular_QAM_modulator_object,randi([0 15],4096,1));    % Generate QAM signal
modulated_baseband_filtered = step(raised_cosine_transmitter_filter,modulatd_baseband);                  % Apply Tx filter
modulated_baseband_filtered_offset = step(phase_frequency_offset_object,modulated_baseband_filtered);                    % Apply frequency offset
received_baseband_noisy = awgn(modulated_baseband_filtered_offset,25,'measured');      % Pass through AWGN channel

%Plot the frequency response of the noisy, frequency-offset signal using the spectrum analyzer. 
%The signal is shifted 250 Hz to the left.
spectrum_analyzer_object.Title = 'Received Signal';
spectrum_analyzer_object(received_baseband_noisy);

%Estimate the frequency offset using frequencyEst. Observe that the estimate is close to the -250 Hz target:
estimated_frequency_offset = step(QAM_coarse_frequency_estimator_object, received_baseband_noisy);

%Correct for the frequency offset using pfoCorrect and the inverse of the estimated frequency offset:
phase_frequency_compensated_baseband = ...
                step(phase_frequency_compensator_object, received_baseband_noisy, -estimated_frequency_offset);

%Plot the frequency response of the compensated signal using the spectrum analyzer. 
%The signal is now properly centered.
spectrum_analyzer_object.Title = 'Frequency-Compensated Signal';
step(spectrum_analyzer_object,phase_frequency_compensated_baseband);




