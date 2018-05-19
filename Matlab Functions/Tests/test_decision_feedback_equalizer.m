
%%%%%%% Transmission Block %%%%%%%
%Set parameters related to the transmission block which is composed of three parts: 
%training sequence, payload, and tail sequence. 
%All three use the same PSK scheme; 
%the training and tail sequences are used for equalization. 
%We use the default random number generator to ensure the repeatability of the results.
symbol_rate = 1e6;  % Symbol rate (Hz)
number_of_training_symbols = 100;  % Number of training symbols
number_of_payload_symbols = 400;  % Number of payload symbols
number_of_tail_symbols = 20;   % Number of tail symbols

%Set random number generator for repeatability:
hStream = RandStream.create('mt19937ar', 'seed', 12345);

%PSK Modulation objects:
%Configure the PSK modulation and demodulation System objects™.
bits_per_symbol = 2; % Number of bits per PSK symbol
modulation_order = 2^bits_per_symbol; % Modulation order
PSK_modulator_object = comm.PSKModulator(modulation_order, ...
                                'PhaseOffset',0, ...
                                'SymbolMapping','Binary');
PSK_demodulator_object = comm.PSKDemodulator(modulation_order, ...
                                'PhaseOffset',0, ...
                                'SymbolMapping','Binary');
PSK_constellation = constellation(PSK_modulator_object).'; % PSK constellation

%Training and Tail Sequences
%Generate the training and tail sequences.
training_data_stream = randi(hStream, [0 modulation_order-1], number_of_training_symbols, 1);
tail_data_stream  = randi(hStream, [0 modulation_order-1], number_of_tail_symbols, 1);
baseband_modulated_training_data = step(PSK_modulator_object,training_data_stream);
baseband_modulated_tail_data = step(PSK_modulator_object,tail_data_stream);

%Transmit and Receive Filters
%Configure raised cosine transmit and receive filter System objects. 
%The filters incorporate upsampling and downsampling, respectively.
raised_cosine_filter_span_in_symbols = 8;  % Filter span in symbols
samples_per_symbol = 4;  % Samples per symbol through channels
raised_cosine_transmitter_filter = comm.RaisedCosineTransmitFilter( ...
                            'RolloffFactor',0.25, ...
                            'FilterSpanInSymbols',raised_cosine_filter_span_in_symbols, ...
                            'OutputSamplesPerSymbol',samples_per_symbol);

raised_cosine_receiver_filter = comm.RaisedCosineReceiveFilter( ...
                            'RolloffFactor',0.25, ...
                            'FilterSpanInSymbols',raised_cosine_filter_span_in_symbols, ...
                            'InputSamplesPerSymbol',samples_per_symbol, ...
                            'DecimationFactor',samples_per_symbol);

%Calculate the samples per symbol after the receive filter:
samples_per_symbol_after_receiver_filter = samples_per_symbol/raised_cosine_receiver_filter.DecimationFactor;
%Calculate the delay in samples from both channel filters:
channel_delay_due_to_receiver_filter = ...
                            raised_cosine_filter_span_in_symbols * samples_per_symbol_after_receiver_filter;

%AWGN Channel:
%Configure an AWGN channel System object with the NoiseMethod property set to 
%Signal to noise ratio (Es/No) and Es/No set to 20 dB.
AWGN_object = comm.AWGNChannel( ...
                            'NoiseMethod','Signal to noise ratio (Es/No)', ...
                            'EsNo',20, ...
                            'SamplesPerSymbol',samples_per_symbol);

%Simulation 1: Linear Equalization for Frequency-Flat Fading
%Begin with single-path, frequency-flat fading channel. 
%For this channel, the receiver uses a simple 1-tap LMS (least mean square) equalizer, 
%which implements automatic gain and phase control.

%The script commadapteqloop.m runs multiple times. 
%Each run corresponds to a transmission block. 
%The equalizer resets its state and weight every transmission block. 
%To retain state from one block to the next, you can set the ResetBeforeFiltering 
%property of the equalizer object to false.

%Before the first run, commadapteqloop.m displays the Rayleigh channel System object and the 
%properties of the equalizer object. 
%For each run, a MATLAB figure shows signal processing visualizations. 
%The red circles in the signal constellation plots correspond to symbol errors. 
%In the "Weights" plot, blue and magenta lines correspond to real and imaginary parts, respectively.
simulation_name = 'Linear equalization for frequency-flat fading';  % Used to label figure window

% Configure a frequency-flat Rayleigh channel System object with the
% RandomStream property set to 'mt19937ar with seed' for repeatability.
rayleigh_channel_object = comm.RayleighChannel( ...
                                    'SampleRate',symbol_rate*samples_per_symbol, ...
                                    'MaximumDopplerShift',30);

% Configure an adaptive equalizer object:
number_of_taps = 1;  % Single weight
LMS_step_size = 0.1; % Step size for LMS algorithm
lms_object = lms(LMS_step_size);  % Adaptive algorithm object
linear_equalizer_object = lineareq(number_of_taps,lms_object,PSK_constellation);  % Equalizer object

%Delay in symbols from the equalizer:
linear_equalizer_delay_in_symbols = (linear_equalizer_object.RefTap-1) / samples_per_symbol_after_receiver_filter;

%Link simulation:
number_of_blocks = 50;  % Number of transmission blocks in simulation
for block = 1:number_of_blocks
    commadapteqloop;
end


%Simulation 2: Linear Equalization for Frequency-Selective Fading
%Simulate a three-path, frequency-selective Rayleigh fading channel. 
%The receiver uses an 8-tap linear RLS (recursive least squares) equalizer with symbol-spaced taps.
simulation_name = 'Linear equalization for frequency-selective fading';

% Reset transmit and receive filters
reset(raised_cosine_transmitter_filter);
reset(raised_cosine_receiver_filter);

% Set the Rayleigh channel System object to be frequency-selective
release(rayleigh_channel_object);
rayleigh_channel_object.PathDelays = [0 0.9 1.5]/symbol_rate;
rayleigh_channel_object.AveragePathGains = [0 -3 -6];

% Configure an adaptive equalizer
number_of_taps = 8;
forgetFactor = 0.99;  % RLS algorithm forgetting factor
lms_object = rls(forgetFactor);  % RLS algorithm object
linear_equalizer_object = lineareq(number_of_taps,lms_object,PSK_constellation);
linear_equalizer_object.RefTap = 3;  % Reference tap
linear_equalizer_delay_in_symbols = (linear_equalizer_object.RefTap-1)/samples_per_symbol_after_receiver_filter;

% Link simulation and store BER values
BERvect = zeros(1,number_of_blocks);
for block = 1:number_of_blocks
    commadapteqloop;
    BERvect(block) = BEREq;
end
avgBER2 = mean(BERvect)

%Simulation 3: Decision feedback Equalization (DFE) for Frequency-Selective Fading
%The receiver uses a DFE with a six-tap fractionally spaced forward filter (two samples per symbol) 
%and two feedback weights. 
%The DFE uses the same RLS algorithm as in Simulation 2. 
%The receive filter structure is reconstructed to account for the increased number of samples per symbol.
simulation_name = 'Decision feedback equalization (DFE) for frequency-selective fading';

% Reset transmit filter and adjust receive filter decimation factor
reset(raised_cosine_transmitter_filter);
release(raised_cosine_receiver_filter);
raised_cosine_receiver_filter.DecimationFactor = 2;
samples_per_symbol_after_receiver_filter = samples_per_symbol/raised_cosine_receiver_filter.DecimationFactor;
channel_delay_due_to_receiver_filter = raised_cosine_filter_span_in_symbols*samples_per_symbol_after_receiver_filter;

% Reset fading channel
reset(rayleigh_channel_object);

% Configure an adaptive equalizer object
number_of_feedforward_weights = 6;  % Number of feedforward equalizer weights
number_of_feedback_weighted = 2;  % Number of feedback filter weights
linear_equalizer_object = dfe(number_of_feedforward_weights, ...
                              number_of_feedback_weighted, ...
                              lms_object, ...
                              PSK_constellation, ...
                              samples_per_symbol_after_receiver_filter);
linear_equalizer_object.RefTap = 3;
linear_equalizer_delay_in_symbols = (linear_equalizer_object.RefTap-1)/samples_per_symbol_after_receiver_filter;

for block = 1:number_of_blocks
    commadapteqloop;
    BERvect(block) = BEREq;
end
avgBER3 = mean(BERvect)


edit('commadapteqloop')
edit('commadapteq_checkvars')
edit('commadapteq_graphics')







