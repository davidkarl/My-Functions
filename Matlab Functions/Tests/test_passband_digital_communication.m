%test passband digital communication


Fc = 2.5e6; %Carrier frequency (Hz)
Fs = 44100; %Sampling frequency [Hz]
symbol_rate = 10; % Symbol rate (symbols/second)
samples_per_symbol = Fs/symbol_rate;  %WHAT HAPPENS IF THE IS NON-INTEGER?!?!?
number_of_symbols_per_frame = 1; % Number of symbols in a frame
modulation_order = 16;             % Modulation order (16-QAM)
baseband_bit_SNR_dB = 15;          % Ratio of baseband bit energy
                    % to noise power spectral density (dB)

% % Calculate sampling frequency in Hz
% Fs = symbol_rate * samples_per_symbol;

% Calculate passband SNR in dB. The noise variance of the baseband signal
% is double that of the corresponding bandpass signal [1]. Increase the SNR
% value by 10*log10(2) dB to account for this difference and have
% equivalent baseband and passband performance.
%DIDN'T THEY MIXED UP WITH THE SECOND TERM?:
passband_SNR_dB = baseband_bit_SNR_dB + 10*log10(log2(modulation_order)/samples_per_symbol) + 10*log10(2);

%Create a constellation diagram for received symbols:
constellation_diagram_object = comm.ConstellationDiagram('SamplesPerSymbol', 1, ...
                                      'XLimits', [-4.5 4.5], ...
                                      'YLimits', [-4.5 4.5], ...
                                      'Position', [70 560 640 460]);


% Create a 16-QAM modulator.
QAM16_modulator_object = comm.RectangularQAMModulator(modulation_order);

%Set the expected constellation of the constellation diagram.
constellation_diagram_object.ReferenceConstellation = constellation(QAM16_modulator_object);

% Generate random data symbols.
symbol_stream = randi([0 modulation_order-1], number_of_symbols_per_frame, 1);

% Modulate the random data.
transmitted_symbol_stream = QAM16_modulator_object(symbol_stream);


% Specify a square root raised cosine filter with a filter length of eight
% symbols and a rolloff factor of 0.2.
filter_length_in_terms_of_symbols = 8;       % Length of the filter in symbols
beta_filter_rolloff_factor = 0.2;     % Rolloff factor

% Design the transmitter filter. Apply a gain to normalize passband gain to unity.
raised_cosine_transmitted_filter = comm.RaisedCosineTransmitFilter(...
    'RolloffFactor', beta_filter_rolloff_factor, ...
    'OutputSamplesPerSymbol', samples_per_symbol, ...
    'FilterSpanInSymbols', filter_length_in_terms_of_symbols, ...
    'Gain', 0.3578);

% Apply pulse shaping by upsampling and filtering.  Alternatively, you can
% use an efficient multirate filter. See help for fdesign.interpolator for
% more information.
transmitted_symbol_stream_filtered = ...
                        raised_cosine_transmitted_filter(transmitted_symbol_stream);

% Plot spectrum estimate of pulse shaped signal.
Fig = figure;
pwelch(transmitted_symbol_stream_filtered,hamming(512),[],[],Fs,'centered')

% Generate carrier. The sqrt(2) factor ensures that the power of the
% frequency upconverted signal is equal to the power of its baseband
% counterpart.
t_vec = (0:1/Fs:(number_of_symbols_per_frame/symbol_rate)-1/Fs).';
carrier = sqrt(2)*exp(1i*2*pi*Fc*t_vec);

% Frequency upconvert to passband.
transmitted_symbol_stream_filtered_upconverted = ...
                    real(transmitted_symbol_stream_filtered .* carrier);

% Plot spectrum estimate.
pwelch(transmitted_symbol_stream_filtered_upconverted,hamming(512),[],[],Fs,'centered')

% Create the passband interference by raising an adjacent channel tone to the third power.
Fc_interference = Fc/3 + 50e3;
interference = 0.7*cos(2*pi*Fc_interference*t_vec+pi/8).^3;

% Calculate the total signal power for the given pulse shape. Account for
% the average power of the baseband 16-QAM upsampled signal. For a
% constellation that contains points with +/- 1 and +/- 3 amplitude levels,
% the average power of a 16-QAM signal is 10 W. The upsampling operation
% reduces this power by a factor of nSamps leading to a net power of
% 10*log10(10/nSamps), or 0.97 dBW for nSamps = 8.
average_power_baseband = 10*log10(sum(abs(constellation(QAM16_modulator_object)).^2)...
                                                              /(modulation_order*samples_per_symbol));
raised_cosine_transmitted_filter_coefficients = coeffs(raised_cosine_transmitted_filter);
signal_power_dB = 10*log10(sum(raised_cosine_transmitted_filter_coefficients.Numerator.^2)) + ...
                                                                                average_power_baseband;

% Add white Gaussian noise based on the computed signal power.
received_symbol_stream_noisy = ...
                awgn(transmitted_symbol_stream_filtered_upconverted, passband_SNR_dB, signal_power_dB);

% Add the adjacent channel interference to the signal.
received_symbol_stream_noisy_intereference = ...
                            received_symbol_stream_noisy + interference;

% Estimate spectrum of the noisy signal and compare it to the spectrum of the original upconverted signal.
figure(Fig);
hold on;
pwelch(received_symbol_stream_noisy_intereference,hamming(512),[],[],Fs,'centered')
ax = Fig.CurrentAxes;
hLines = ax.Children;
hLines(1).Color = [1 0 0];
legend('Signal at channel input',...
    'Signal at channel output','Location','southwest')

% Downconvert to baseband (Assumes perfect synchronization).
received_symbol_stream_noisy_interference_downconverted = ...
                        received_symbol_stream_noisy_intereference .* conj(carrier);

% Estimate spectrum of the downconverted signal with adjacent channel interference.
figure(Fig);
hold off;
pwelch(received_symbol_stream_noisy_interference_downconverted,hamming(512),[],[],Fs,'centered')

% Design the receive filter. Apply a gain to normalize passband gain to
% unity.
raised_cosine_receiver_filter = comm.RaisedCosineReceiveFilter(...
  'RolloffFactor', raised_cosine_transmitted_filter.RolloffFactor, ...
  'InputSamplesPerSymbol', raised_cosine_transmitted_filter.OutputSamplesPerSymbol, ...
  'FilterSpanInSymbols', raised_cosine_transmitted_filter.FilterSpanInSymbols, ...
  'DecimationFactor', 1, ...
  'Gain', 0.3578);

% Filter the frequency downconverted signal.
filtered_received_downconverted_symbol_stream = ...
                   raised_cosine_receiver_filter(received_symbol_stream_noisy_interference_downconverted);

% Estimate spectrum of the filtered signal and compare it to the spectrum
% of the signal at the filter input.
figure(Fig);
hold on;
pwelch(filtered_received_downconverted_symbol_stream,hamming(512),[],[],Fs,'centered');
ax = Fig.CurrentAxes;
hLines = ax.Children;
hLines(1).Color = [1 0 0];
legend('Signal at filter input',...
    'Signal at filter output','Location','southwest')

% Amplify the signal to compensate for the power loss caused by pulse
% shaping and matched filtering. This places the received signal symbols
% around the expected 16-QAM constellation points.
filtered_received_downconverted_symbol_stream = filter_length_in_terms_of_symbols*filtered_received_downconverted_symbol_stream;

% Downsample the filtered signal. Discard the first nSym symbols due to filter delay.
e2e_delay = samples_per_symbol * filter_length_in_terms_of_symbols;
filtered_received_downconverted_symbol_stream = ...
                    filtered_received_downconverted_symbol_stream(e2e_delay+1:samples_per_symbol:end);

% Obtain the constellation diagram of the received signal with adjacent channel interference.
constellation_diagram_object(filtered_received_downconverted_symbol_stream)

% Create demodulator.
QAM16_demodulator_object = comm.RectangularQAMDemodulator(modulation_order);

% Demodulate received symbols and count the number of symbol errors.
received_demodulated_symbols = QAM16_demodulator_object(filtered_received_downconverted_symbol_stream);
number_of_errors = ...
                sum(received_demodulated_symbols~=symbol_stream(1:end-filter_length_in_terms_of_symbols));

%without interference:
y = received_symbol_stream_noisy.*conj(carrier);
rcvSym = raised_cosine_receiver_filter(y);
rcvSym = filter_length_in_terms_of_symbols*rcvSym;
rcvSymDown = rcvSym(e2e_delay+1:samples_per_symbol:end);
scatScope2 = clone(constellation_diagram_object);
hide(constellation_diagram_object);
% Update the constellation diagram for the interference-free received symbols.
scatScope2(rcvSymDown)

xHat = QAM16_demodulator_object(rcvSymDown);
number_of_errors = sum(xHat~=symbol_stream(1:end-filter_length_in_terms_of_symbols));













