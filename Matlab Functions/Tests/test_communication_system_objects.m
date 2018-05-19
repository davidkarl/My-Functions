%test communication system object

%Sampling / Sound parameters:
Fs = 44100;
Fc = 15000;

%symbols / frame parameters:
symbol_rate = 1; %[Hz]
samples_per_symbol = Fs/symbol_rate;
number_of_symbols_per_frame = 1;
samples_per_frame = samples_per_symbol*number_of_symbols_per_frame;
%FSK parameters:
number_of_FSK_frequencies = 2;
frequency_seperation_in_Hz = 2000; 
%Create FSK modulation and demodulation objects:
FSK_modulation_object = comm.FSKModulator(number_of_FSK_frequencies,frequency_seperation_in_Hz,'SamplesPerSymbol',samples_per_symbol,'SymbolRate',symbol_rate);
FSK_demodulation_object = comm.FSKDemodulator(number_of_FSK_frequencies,frequency_seperation_in_Hz,'SamplesPerSymbol',samples_per_symbol,'SymbolRate',symbol_rate);
%Create white noise generation object:
WGN_SNR_in_dB = inf;
white_noise_generation_object = comm.AWGNChannel('NoiseMethod', ...
    'Signal to noise ratio (SNR)','SNR',WGN_SNR_in_dB);
%Create error rate object:
error_rate_object = comm.ErrorRate;


%Spectrum Analyzer / Sound stuff:
t_vec = my_linspace(0,1/Fs,samples_per_frame);
delta_t = samples_per_frame / Fs;
phase_vec = mod(2*pi*Fc*t_vec,2*pi);
spectrum_analyzer = dsp.SpectrumAnalyzer('SampleRate',Fs,...
                                         'ShowLegend',false,...
                                         'Window','Rectangular',...
                                         'SpectralAverages',1,...
                                         'WindowLength',samples_per_frame);
                                     
%Analytic signal object:
analytic_signal_object = dsp.AnalyticSignal;

%Audio player object:
queue_duration = 1;
audio_player_object = dsp.AudioPlayer('SampleRate',Fs);
audio_player_object.QueueDuration = queue_duration;
                                     
%Transmit one hundred 50-symbol frames using FSK in an AWGN channel.
number_of_frames = 100;
for counter = 1:number_of_frames
    binary_data = randi([0,number_of_FSK_frequencies-1],number_of_symbols_per_frame,1);
    modulated_signal = step(FSK_modulation_object,binary_data);
    
    
    carrier_signal = exp(-1i*phase_vec);
    upconverted_signal = real( modulated_signal .* carrier_signal);
    
    step(audio_player_object,upconverted_signal);
    
    analytic_signal_received = step(analytic_signal_object,upconverted_signal);
    downconverted_signal_received = (analytic_signal_received .* conj(carrier_signal));
    
    demodulated_data = step(FSK_demodulation_object,downconverted_signal_received);
    error_stats = step(error_rate_object,binary_data,demodulated_data);
    
    step(spectrum_analyzer , upconverted_signal);
    
    phase_vec = mod(phase_vec + 2*pi*Fc*delta_t,2*pi);
    
end

es = 'Error rate = %4.2e\nNumber of errors = %d\nNumber of symbols = %d\n';
fprintf(es,error_stats)













