%test_pulse_amplitude_modulation_communication

%Basic parameters (Assuming Fs given, and i insist on using communication toolbox):
Fs = 44100;
Fc = 4000;
symbol_rate = 10; %[symbols/second]
samples_per_symbol = round(Fs/symbol_rate); %which must be integer as of now with communication toolbox
symbol_rate = Fs/samples_per_symbol;

%Number of frequencies/amplitudes/constallation points:
modulation_order = 2; %In the context of QAM and PAM it's the number of constallation points

%Frame paramters:
number_of_symbols_per_frame = 5;
samples_per_frame = number_of_symbols_per_frame * samples_per_symbol;
t_vec = my_linspace(0,1/Fs,samples_per_frame);
delta_t = samples_per_frame / Fs;
phase_vec_transmitter = mod(2*pi*Fc*t_vec,2*pi);
phase_vec_receiver = mod(2*pi*Fc*t_vec + 0*randn(1), 2*pi);

%SNR (baseband):
baseband_bit_SNR_dB = 15; %Ratio of baseband bit energy

%Create a 16-PAM modulator System object with bits as inputs and Gray-coded signal constellation
PAM_modulator_object = comm.PAMModulator(modulation_order,'BitInput',false,'SymbolMapping','Gray');
PAM_demodulator_object = comm.PAMDemodulator(modulation_order);
AWGN_object = comm.AWGNChannel('NoiseMethod', 'Signal to noise ratio (SNR)', 'SNR',inf);
error_rate_object = comm.ErrorRate;


%Plot constellation and create constellation object:
constellation(PAM_modulator_object);
constellation_diagram_object = comm.ConstellationDiagram('SamplesPerSymbol', 1, ...
                                                         'XLimits', [-4.5 4.5], ...
                                                         'YLimits', [-4.5 4.5], ...
                                                         'Position', [70 560 640 460]);

%Create transmitter and receiver raised cosine filter:
%- Specify a square root raised cosine filter with a filter length of eight
%- symbols and a rolloff factor of 0.2.
%- Design the transmitter filter. Apply a gain to normalize passband gain to unity.
filter_length_in_terms_of_symbols = 8;       % Length of the filter in symbols
beta_filter_rolloff_factor = 0.2;     % Rolloff factor
raised_cosine_transmitted_filter = comm.RaisedCosineTransmitFilter(...
    'RolloffFactor', beta_filter_rolloff_factor, ...
    'OutputSamplesPerSymbol', samples_per_symbol, ...
    'FilterSpanInSymbols', filter_length_in_terms_of_symbols, ...
    'Gain', 0.3578);
raised_cosine_receiver_filter = comm.RaisedCosineReceiveFilter(...
    'RolloffFactor', raised_cosine_transmitted_filter.RolloffFactor, ...
    'InputSamplesPerSymbol', raised_cosine_transmitted_filter.OutputSamplesPerSymbol, ...
    'FilterSpanInSymbols', raised_cosine_transmitted_filter.FilterSpanInSymbols, ...
    'DecimationFactor', 1, ...
    'Gain', 0.3578);


%Spectrum Analyzer:
spectrum_analyzer = dsp.SpectrumAnalyzer('SampleRate',Fs,...
                                         'ShowLegend',false,...
                                         'Window','Rectangular',...
                                         'SpectralAverages',1,...
                                         'WindowLength',samples_per_frame);
                                     
%Analytic signal object:
analytic_signal_object = dsp.AnalyticSignal;

%Variable Fractional Delay object:
variable_fractional_delay_object = dsp.VariableFractionalDelay;
samples_per_symbol_fraction = abs(randn(1));
number_of_samples_delay = samples_per_symbol*samples_per_symbol_fraction;
 
for frame_counter = 1:100 
    %Get a symbol frame:
    current_symbol_stream = randi([0 PAM_modulator_object.ModulationOrder-1],number_of_symbols_per_frame,1);
    modulated_baseband = step(PAM_modulator_object, current_symbol_stream);
    %PAM modulator object doesn't have "SamplesPerSymbol" so i need to replicate baseband:
    [modulated_baseband_upsampled] = upsample_signal(modulated_baseband,samples_per_symbol);
     
    %Upconvert and Transmitt:
    phase_vec_transmitter = mod(phase_vec_transmitter + 2*pi*Fc*delta_t, 2*pi);
    transmitter_carrier = exp(-1i*phase_vec_transmitter);
    modulated_carrier = modulated_baseband_upsampled .* transmitter_carrier;
    
    %Add noise if wanted:
    modulated_carrier_received = step(AWGN_object, (modulated_carrier)); %how is AWGN add noise to complex signal?
    
    %Show transmitted carrier:
    figure('units','normalized','outerposition',[0 0 0.9 0.9])
    subplot(4,1,1)
    plot(real(modulated_baseband_upsampled));
    title('modulated baseband I (real) component');
    subplot(4,1,2)
    plot(real(transmitter_carrier));
    title('transmitter carrier (real)');
    subplot(4,1,3)
    plot(real(modulated_carrier));
    title('modulated carrier transmitted (real)');
    subplot(4,1,4)
    plot(real(modulated_carrier_received)); 
    title('modulated carrier received (real)');
    close(gcf); %close current graph
    
    %Add delay:
    signal_received_delayed = step(variable_fractional_delay_object, modulated_carrier_received, samples_per_symbol_fraction);
    
    %Downconvert:
    phase_vec_receiver = mod(phase_vec_receiver + 2*pi*Fc*delta_t, 2*pi);
    receiver_carrier = exp(-1i*phase_vec_receiver);
    signal_received_downconverted = signal_received_delayed .* conj(receiver_carrier);
     
    %Demodulate (decide on symbols for highly sampled received, downconverted signal):
    demodulated_signal_upsampled = step(PAM_demodulator_object, (signal_received_downconverted));
    [demodulated_signal_averaged] = average_signal_every_n_samples(demodulated_signal_upsampled,samples_per_symbol);
    errorStats = step(error_rate_object, current_symbol_stream, demodulated_signal_averaged);
end
fprintf('Error rate = %f\nNumber of errors = %d\n', ...
    errorStats(1), errorStats(2))

 








