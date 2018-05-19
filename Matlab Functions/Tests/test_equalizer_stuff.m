
% Build a set of test data.
BPSK_modulator_object = comm.BPSKModulator; % BPSKModulator System object
baseband_modulated_symbol_stream = step(BPSK_modulator_object,randi([0 1],1000,1)); % BPSK symbols
distorted_baseband_modulated_symbol_stream = conv(baseband_modulated_symbol_stream,[1 0.8 0.3]);  % Received signal
% Create an equalizer object.
linear_equalizer_lms_object = lineareq(8,lms(0.03));
% Change the reference tap index in the equalizer.
linear_equalizer_lms_object.RefTap = 4;
% Apply the equalizer object to a signal.
baseband_modulated_symbol_stream_distorted_equalized = equalize(linear_equalizer_lms_object,...
                                   distorted_baseband_modulated_symbol_stream,...
                                   baseband_modulated_symbol_stream(1:200));



%Create equalizer objects of different types:
linear_equalizer_rls = lineareq(10,rls(0.3)); % Symbol-spaced linear
linear_equalizer_rls_fractionally_spaced = lineareq(10,rls(0.3),[-1 1],2); % Fractionally spaced linear
linear_equalizer_decision_feedback_equalizer_rls = dfe(3,2,rls(0.3)); % DFE
linear_equalizer_variable_size_lms = lineareq(10,varlms(0.01,0.01,0,1)); % Create equalizer object.
linear_equalizer_variable_size_lms.nWeights = 8; % Change the number of weights from 10 to 8.
% MATLAB automatically changes the sizes of eqlvar.Weights and
% eqlvar.WeightInputs.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% The following code illustrates how to use equalize with a training sequence. 
%The training sequence in this case is just the beginning of the transmitted message.

% Set up parameters and signals.
modulation_order = 4; % Alphabet size for modulation
symbol_stream = randi([0 modulation_order-1],1500,1); % Random message
QPSK_modulator_object = comm.QPSKModulator('PhaseOffset',0);
baseband_modulated_symbol_stream = step(QPSK_modulator_object,symbol_stream); % Modulate using QPSK.
training_sequence_length = 500; % Length of training sequence
distortion_filter_coefficients = [.986; .845; .237; .123+.31i]; % Channel coefficients
distorted_baseband_modulated_symbol_stream = filter(distortion_filter_coefficients,1,baseband_modulated_symbol_stream); % Introduce channel distortion.

% Equalize the received signal.
linear_equalizer_lms_object = lineareq(8, lms(0.01)); % Create an equalizer object.
linear_equalizer_lms_object.SigConst = step(QPSK_modulator_object,(0:modulation_order-1)')'; % Set signal constellation.
[symbolest,y_detected] = equalize(linear_equalizer_lms_object,...
                          distorted_baseband_modulated_symbol_stream,...
                          baseband_modulated_symbol_stream(1:training_sequence_length)); % Equalize.

% Plot signals.
h = scatterplot(distorted_baseband_modulated_symbol_stream,1,training_sequence_length,'bx'); 
hold on;
scatterplot(symbolest,1,training_sequence_length,'g.',h);
scatterplot(linear_equalizer_lms_object.SigConst,1,0,'k*',h);
legend('Filtered signal','Equalized signal',...
   'Ideal signal constellation');
hold off;

% Compute error rates with and without equalization.
QPSK_demodulator_object = comm.QPSKDemodulator('PhaseOffset',0);
demodulated_stream_without_equalizer = step(QPSK_demodulator_object,distorted_baseband_modulated_symbol_stream); % Demodulate unequalized signal.
demodulated_stream_with_equalizer = step(QPSK_demodulator_object,y_detected); % Demodulate detected signal from equalizer.
error_rate_object = comm.ErrorRate; % ErrorRate calculator
symbol_error_rate_without_equalizer = step(error_rate_object, ...
                baseband_modulated_symbol_stream(training_sequence_length+1:end), ...
                demodulated_stream_without_equalizer(training_sequence_length+1:end));
reset(error_rate_object)
symbol_error_rate_with_equalizer = step(error_rate_object, ...
                                        baseband_modulated_symbol_stream(training_sequence_length+1:end), ...
                                        demodulated_stream_with_equalizer(training_sequence_length+1:end));
disp('Symbol error rates with and without equalizer:')
disp([symbol_error_rate_with_equalizer(1) symbol_error_rate_without_equalizer(1)])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Example: Equalizing Multiple Times, Varying the Mode  
%If you invoke equalize multiple times with the same equalizer object to equalize a series of signal vectors, 
%you might use a training sequence the first time you call the function and omit the training sequence in 
%subsequent calls. 
%Each iteration of the equalize function after the first one operates completely in decision-directed mode. 
%However, because the ResetBeforeFiltering property of the equalizer object is set to 0, the equalize function 
%uses the existing state information in the equalizer object when starting each iteration's equalization operation. 
%As a result, the training affects all equalization operations, not just the first.

%The code below illustrates this approach. 
%Notice that the first call to equalize uses a training sequence as an input argument, 
%and the second call to equalize omits a training sequence.

modulation_order = 4; % Alphabet size for modulation
symbol_stream = randi([0 modulation_order-1],1500,1); % Random message
QPSK_modulator_object = comm.QPSKModulator('PhaseOffset',0);
baseband_modulated_symbol_stream = step(QPSK_modulator_object,symbol_stream); % Modulate using QPSK.
training_length = 500; % Length of training sequence
distortion_filter_coefficients = [.986; .845; .237; .123+.31i]; % Channel coefficients
distorted_baseband_modulated_symbol_stream = ...
                                    filter(distortion_filter_coefficients,1,baseband_modulated_symbol_stream); % Introduce channel distortion.

% Set up equalizer.
linear_equalizer_lms_object = lineareq(8, lms(0.01)); % Create an equalizer object.
linear_equalizer_lms_object.SigConst = step(QPSK_modulator_object,(0:modulation_order-1)')'; % Set signal constellation.
% Maintain continuity between calls to equalize.
linear_equalizer_lms_object.ResetBeforeFiltering = 0;

% Equalize the received signal, in pieces.
% 1. Process the training sequence.
s1 = equalize(linear_equalizer_lms_object,distorted_baseband_modulated_symbol_stream(1:training_length),baseband_modulated_symbol_stream(1:training_length));
% 2. Process some of the data in decision-directed mode.
s2 = equalize(linear_equalizer_lms_object,distorted_baseband_modulated_symbol_stream(training_length+1:800));
% 3. Process the rest of the data in decision-directed mode.
s3 = equalize(linear_equalizer_lms_object,distorted_baseband_modulated_symbol_stream(801:end));
baseband_modulated_symbol_stream_distorted_equalized = [s1; s2; s3]; % Full output of equalizer


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%Delays from Equalization.  
%(*) For proper equalization using adaptive algorithms other than CMA, you should set the reference tap so 
%that it exceeds the delay, in symbols, between the transmitter's modulator output and the equalizer input. 
%When this condition is satisfied, the total delay between the modulator output and the equalizer output 
%is equal to (RefTap-1)/nSampPerSym symbols. 
%(*) Because the channel delay is typically unknown, a common practice is to set the reference tap to 
%the center tap in a linear equalizer, or the center tap of the forward filter in a decision-feedback equalizer.
%(*) For CMA equalizers, the expression above does not apply because a CMA equalizer has no reference tap. 
%If you need to know the delay, you can find it empirically after the equalizer weights have converged. 
%Use the xcorr function to examine cross-correlations of the modulator output and the equalizer output.

%Techniques for Working with Delays  
%Here are some typical ways to take a delay of D into account by padding or truncating data:
%Pad your original data with D extra symbols at the end. 
%Before comparing the original data with the received data, omit the first D symbols of the received data. 
%In this approach, all the original data (not including the padding) is accounted for in the received data.
%Before comparing the original data with the received data, omit the last D symbols of the original data and 
%the first D symbols of the received data. 
%In this approach, some of the original symbols are not accounted for in the received data.
%The example below illustrates the latter approach. 
%For an example that illustrates both approaches in the context of interleavers, 
%see Delays of Convolutional Interleavers.

modulation_order = 2; % Use BPSK modulation for this example.
symbol_stream = randi([0 modulation_order-1],1000,1); % Random data
BPSK_modulator_object = comm.BPSKModulator('PhaseOffset',0);
baseband_modulated_symbol_stream = step(BPSK_modulator_object,symbol_stream); % Modulate
training_length = 100; % Length of training sequence
training_baseband_modulated_symbol_stream = baseband_modulated_symbol_stream(1:training_length); % Training sequence

% Define an equalizer and equalize the received signal.
linear_equalizer_normalized_lms = lineareq(3,normlms(.0005,.0001),pskmod(0:modulation_order-1,modulation_order));
linear_equalizer_normalized_lms.RefTap = 2; % Set reference tap of equalizer.
[equalized_signal,determined_symbols] = equalize(linear_equalizer_normalized_lms,baseband_modulated_symbol_stream,training_baseband_modulated_symbol_stream); % Equalize.

BPSK_demodulator_object = comm.BPSKDemodulator('PhaseOffset',0);
demodulated_massage = step(BPSK_demodulator_object,determined_symbols); % Demodulate the detected signal.

% Compute bit error rate while compensating for delay introduced by RefTap
% and ignoring training sequence.
D = (linear_equalizer_normalized_lms.RefTap -1)/linear_equalizer_normalized_lms.nSampPerSym;
error_rate_object = comm.ErrorRate('ReceiveDelay',D);
berVec = step(error_rate_object, symbol_stream(training_length+1:end), demodulated_massage(training_length+1:end));
ber = berVec(1);
numerrs = berVec(2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Equalize Using a Loop.  
%If your data is partitioned into a series of vectors (that you process within a loop, for example), 
%you can invoke the equalize function multiple times, saving the equalizer's internal state information 
%for use in a subsequent invocation. 
%In particular, the final values of the WeightInputs and Weights properties in one equalization 
%operation should be the initial values in the next equalization operation. 
%This section gives an example, followed by more general procedures for equalizing within a loop.

%Example: Adaptive Equalization Within a Loop  
%The example below illustrates how to use equalize within a loop, varying the equalizer between iterations. 
%Because the example is long, this discussion presents it in these steps:

%If you want to equalize iteratively while potentially changing equalizers between iterations, 
%see Changing the Equalizer Between Iterations for help generalizing from this example to other cases.

%Initializing Variables  
%The beginning of the example defines parameters and creates three equalizer objects:
%An RLS equalizer object.
%An LMS equalizer object.
%A variable, eq_current, that points to the equalizer object to use in the current iteration of the loop. 
%Initially, this points to the RLS equalizer object. 
%After the second iteration of the loop, eq_current is redefined to point to the LMS equalizer object.

%Set up parameters.
modulation_order = 16; % Alphabet size for modulation
signal_constellation = step(comm.RectangularQAMModulator(modulation_order),(0:modulation_order-1)');
                                    % Signal constellation for 16-QAM
distortion_filter_coefficients = [1 0.45 0.3+0.2i]; % Channel coefficients
rectangular_QAM_modulator = comm.RectangularQAMModulator(modulation_order); % QAMModulator System object

%Set up equalizers:
linear_equalizer_rls_object = lineareq(6, rls(0.99,0.1)); % Create an RLS equalizer object.
linear_equalizer_rls_object.SigConst = signal_constellation'; % Set signal constellation.
linear_equalizer_rls_object.ResetBeforeFiltering = 0; % Maintain continuity between iterations.
linear_equalizer_lms_object = lineareq(6, lms(0.003)); % Create an LMS equalizer object.
linear_equalizer_lms_object.SigConst = signal_constellation'; % Set signal constellation.
linear_equalizer_lms_object.ResetBeforeFiltering = 0; % Maintain continuity between iterations.
current_linear_equalizer = linear_equalizer_rls_object; % Point to RLS for first iteration.

%Simulating the System Using a Loop:  
%The next portion of the example is a loop that Generates a signal to transmit and selects a portion to 
%use as a training sequence in the first iteration of the loop
%Introduces channel distortion
%Equalizes the distorted signal using the chosen equalizer for this iteration, 
%retaining the final state and weights for later use 
%Plots the distorted and equalized signals, for comparison
%Switches to an LMS equalizer between the second and third iterations

%Main loop
for loop_counter = 1:4
   symbol_stream = randi([0 modulation_order-1],500,1); % Random message
   baseband_modulated_symbol_stream = step(rectangular_QAM_modulator,symbol_stream); % Modulate using 16-QAM.

   %Set up training sequence for first iteration:
   if loop_counter == 1
      training_sequence_length = 200; 
      training_signal = baseband_modulated_symbol_stream(1:training_sequence_length);
   else
      % Use decision-directed mode after first iteration.
      training_sequence_length = 0; 
      training_signal = [];
   end

   %Introduce channel distortion:
   baseband_modulated_symbol_stream_distorted = ...
                                        filter(distortion_filter_coefficients,1,baseband_modulated_symbol_stream);

   %Equalize the received signal:
   baseband_modulated_symbol_stream_distorted_equalized = ...
                     equalize(current_linear_equalizer,baseband_modulated_symbol_stream_distorted,training_signal);

   %Plot signals:
   h = scatterplot(baseband_modulated_symbol_stream_distorted(training_sequence_length+1:end),1,0,'bx'); hold on;
   scatterplot(baseband_modulated_symbol_stream_distorted_equalized(training_sequence_length+1:end),1,0,'g.',h);
   scatterplot(signal_constellation,1,0,'k*',h);
   legend('Received signal','Equalized signal','Signal constellation');
   title(['Iteration #' num2str(loop_counter) ' (' current_linear_equalizer.AlgType ')']);
   hold off;

   % Switch from RLS to LMS after second iteration.
   if loop_counter == 2
      linear_equalizer_lms_object.WeightInputs = current_linear_equalizer.WeightInputs; % Copy final inputs.
      linear_equalizer_lms_object.Weights = current_linear_equalizer.Weights; % Copy final weights.
      current_linear_equalizer = linear_equalizer_lms_object; % Make eq_current point to eqlms.
   end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%MLSE Equalizers
%Equalizing a Vector Signal
%In its simplest form, the mlseeq function equalizes a vector of modulated data when you specify the 
%estimated coefficients of the channel (modeled as an FIR filter), the signal constellation for the modulation type, 
%and the traceback depth that you want the Viterbi algorithm to use. 
%Larger values for the traceback depth can improve the results from the equalizer but increase the computation time.

%An example of the basic syntax for mlseeq is below.
modulation_order = 4; 
QPSK_modulator_object = comm.QPSKModulator;
signal_constellation = step(QPSK_modulator_object,(0:modulation_order-1)'); % 4-PSK constellation
baseband_modulated_symbol_stream = step(QPSK_modulator_object,[1 2 2 0 3 1 3 3 2 1 0 2 3 0 1]'); % Modulated message
distortion_filter_coefficients = [.986; .845; .237; .12345+.31i]; % Channel coefficients
baseband_modulated_symbol_stream_distorted = filter(distortion_filter_coefficients,1,baseband_modulated_symbol_stream); % Introduce channel distortion.
traceback_depth =  10; % Traceback depth for equalizer
channel_estimate = distortion_filter_coefficients; % Assume the channel is known exactly.
MLSE_equalizer_object = comm.MLSEEqualizer('TracebackDepth',traceback_depth,...
                                           'Channel',channel_estimate, ...
                                           'Constellation',signal_constellation);
baseband_modulated_symbol_stream_distorted_equalized = step(MLSE_equalizer_object,baseband_modulated_symbol_stream_distorted); % Equalize.

%The mlseeq function has two operation modes:
%Continuous operation mode: enables you to process a series of vectors using repeated calls to mlseeq, 
%where the function saves its internal state information from one call to the next. 
%To learn more, see Equalizing in Continuous Operation Mode.
%Reset operation mode: enables you to specify a preamble and postamble that accompany your data. 
%To learn more, see Use a Preamble or Postamble.
%If you are not processing a series of vectors and do not need to specify a preamble or postamble, 
%the operation modes are nearly identical. 
%However, they differ in that continuous operation mode incurs a delay, while reset operation mode does not. 
%The example above could have used either mode, except that substituting continuous operation mode 
%would have produced a delay in the equalized output. 
%To learn more about the delay in continuous operation mode, see Delays in Continuous Operation Mode.

%Equalizing in Continuous Operation Mode
%If your data is partitioned into a series of vectors (that you process within a loop, for example), 
%continuous operation mode is an appropriate way to use the mlseeq function. 
%In continuous operation mode, mlseeq can save its internal state information for use in a subsequent 
%invocation and can initialize using previously stored state information. 
%To choose continuous operation mode, use 'cont' as an input argument when invoking mlseeq.

%Note: Continuous operation mode incurs a delay, as described in Delays in Continuous Operation Mode. 
%Also, continuous operation mode cannot accommodate a preamble or postamble.
%Procedure for Continuous Operation Mode.  
%The typical procedure for using continuous mode within a loop is as follows:

%Before the loop starts, create three empty matrix variables (for example, sm, ts, ti) that eventually 
%store the state metrics, traceback states, and traceback inputs for the equalizer.
%Inside the loop, invoke mlseeq using a syntax like
% [equalized_symbol_stream,sm,ts,ti] = ...
%     mlseeq(x,distortion_filter_coefficients,signal_constellation,traceback_depth,'cont',nsamp,sm,ts,ti);
%Using sm, ts, and ti as input arguments causes mlseeq to continue from where it finished in the previous iteration. 
%Using sm, ts, and ti as output arguments causes mlseeq to update the state information at the end of the 
%current iteration. 
%In the first iteration, sm, ts, and ti start as empty matrices, so the first invocation of the mlseeq 
%function initializes the metrics of all states to 0.
%Delays in Continuous Operation Mode.  
%Continuous operation mode with a traceback depth of tblen incurs an output delay of tblen symbols. 
%This means that the first tblen output symbols are unrelated to the input signal, while the last 
%tblen input symbols are unrelated to the output signal. 
%For example, the command below uses a traceback depth of 3, and the first 3 output symbols are unrelated 
%to the input signal of ones(1,10).

baseband_modulated_symbol_stream_distorted_equalized = step(comm.MLSEEqualizer('Channel',1, ...
                                                  'Constellation',[-7:2:7], ...
                                                  'TracebackDepth',3,...
                                                  'TerminationMethod', 'Continuous'), ...
                               complex(ones(10,1)));
% y =
% 
%     -7   -7   -7    1    1    1    1    1    1    1

%Keeping track of delays from different portions of a communication system is important, 
%especially if you compare signals to compute error rates. 
%The example in Example: Continuous Operation Mode illustrates how to take the delay into account 
%when computing an error rate.

%Example: Continuous Operation Mode.  
%The example below illustrates the procedure for using continuous operation mode within a loop. 
%Because the example is long, this discussion presents it in multiple steps:

%Initializing Variables  
%The beginning of the example defines parameters, initializes the state variables sm, ts, and ti, and initializes variables that accumulate results from each iteration of the loop.

number_of_symbols_per_frame = 200; % Number of symbols in each iteration
number_of_iterations = 25; % Number of iterations
modulation_order = 4; % Use 4-PSK modulation.
QPSK_modulator_object = comm.QPSKModulator('PhaseOffset',0);
signal_constellation = step(QPSK_modulator_object,(0:modulation_order-1)'); % PSK constellation
distortion_filter_coefficients = [1 ; 0.25]; % Channel coefficients
channel_estimate = distortion_filter_coefficients; % Channel estimate
traceback_depth = 10; % Traceback depth for equalizer
state_metrics = []; 
traceback_states = []; 
traceback_inputs = []; % Initialize equalizer data.
%Initialize cumulative results:
fullmodmsg = []; 
fullfiltmsg = []; 
fullrx = [];
MLSE_equalizer_object = comm.MLSEEqualizer('TracebackDepth',traceback_depth, ...
                                           'Channel', channel_estimate, ...
                                           'Constellation',signal_constellation, ...
                                           'TerminationMethod', 'Continuous');
%Simulating the System Using a Loop  
%The middle portion of the example is a loop that generates random data, 
%modulates it using baseband PSK modulation, and filters it. 
%Finally, mlseeq equalizes the filtered data. 
%The loop also updates the variables that accumulate results from each iteration of the loop.
for iteration_counter = 1:number_of_iterations
    symbol_stream = randi([0 modulation_order-1],number_of_symbols_per_frame,1); % Random signal vector
    baseband_modulated_symbol_stream = step(QPSK_modulator_object,symbol_stream); % PSK-modulated signal
    baseband_modulated_symbol_stream_distorted = filter(distortion_filter_coefficients,1,baseband_modulated_symbol_stream); % Filtered signal
    equalized_stream = step(MLSE_equalizer_object,baseband_modulated_symbol_stream_distorted); % Equalize
    
    %Update vectors with cumulative results:
    fullmodmsg = [fullmodmsg; baseband_modulated_symbol_stream];
    fullfiltmsg = [fullfiltmsg; baseband_modulated_symbol_stream_distorted];
    fullrx = [fullrx; equalized_stream];
end

%Computing an Error Rate and Plotting Results
%The last portion of the example computes the symbol error rate from all iterations of the loop. 
%The symerr function compares selected portions of the received and transmitted signals, not the entire signals. 
%Because continuous operation mode incurs a delay whose length in samples is the traceback depth (tblen) 
%of the equalizer, it is necessary to exclude the first tblen samples from the received signal and the last 
%tblen samples from the transmitted signal. 
%Excluding samples that represent the delay of the equalizer ensures that the symbol error rate 
%calculation compares samples from the received and transmitted signals that are meaningful and 
%that truly correspond to each other.

%The example also plots the signal before and after equalization in a scatter plot. 
%The points in the equalized signal coincide with the points of the ideal signal constellation for 4-PSK.

%Compute total number of symbol errors. Take the delay into account.
error_rate_object = comm.ErrorRate('ReceiveDelay',10);
err = step(error_rate_object, fullmodmsg, fullrx);
number_of_symbol_errors = err(1);

% Plot signal before and after equalization.
h = scatterplot(fullfiltmsg); 
hold on;
scatterplot(fullrx,1,0,'r*',h);
legend('Filtered signal before equalization','Equalized signal',...
       'Location','NorthOutside');
hold off;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Use a Preamble or Postamble
%Some systems include a sequence of known symbols at the beginning or end of a set of data. 
%The known sequence at the beginning or end is called a preamble or postamble, respectively. 
%The mlseeq function can accommodate a preamble and postamble that are already incorporated into its input signal. 
%When you invoke the function, you specify the preamble and postamble as integer vectors that represent 
%the sequence of known symbols by indexing into the signal constellation vector. 
%For example, a preamble vector of [1 4 4] and a 4-PSK signal constellation of [1 j -1 -j] indicate 
%that the modulated signal begins with [1 -j -j].

%If your system uses a preamble without a postamble, use a postamble vector of [] when invoking mlseeq. 
%Similarly, if your system uses a postamble without a preamble, use a preamble vector of [].

%Use a Preamble in MATLAB  
%The example below illustrates how to accommodate a preamble when using mlseeq. 
%The same preamble symbols appear at the beginning of the message vector and in the syntax for mlseeq. 
%If you want to use a postamble, you can append it to the message vector and also include it as the last 
%input argument for mlseeq. 
%In this example, however, the postamble input in the mlseeq syntax is an empty vector because the 
%system uses no postamble.

modulation_order = 4; 
QPSK_modulator_object = comm.QPSKModulator;% Use 4-PSK modulation.
signal_constellation = step(QPSK_modulator_object,(0:modulation_order-1)'); % PSK constellation
traceback_depth = 16; % Traceback depth for equalizer

preamble = [3; 1]; % Expected preamble, as integers
symbol_stream = randi([0 modulation_order-1],98,1); % Random symbols
symbol_stream = [preamble; symbol_stream]; % Include preamble at the beginning.
baseband_modulated_symbol_stream = step(QPSK_modulator_object,symbol_stream); % Modulated message
distortion_filter_coefficients = [.623; .489+.234i; .398i; .21]; % Channel coefficients
channel_estimate = distortion_filter_coefficients; % Channel estimate
MLSE_equalizer_object = comm.MLSEEqualizer('TracebackDepth',traceback_depth,...
                            'Channel',channel_estimate, ...
                            'Constellation',signal_constellation, ...
                            'PreambleSource', 'Property', ...
                            'Preamble', preamble);
baseband_modulated_symbol_stream_distorted = ...
                filter(distortion_filter_coefficients,1,baseband_modulated_symbol_stream); % Introduce channel distortion.
baseband_modulated_symbol_stream_distorted_equalized = ...
                        step(MLSE_equalizer_object,baseband_modulated_symbol_stream_distorted);

%Symbol error rate:
error_rate_object = comm.ErrorRate;
serVec  = step(error_rate_object, baseband_modulated_symbol_stream,baseband_modulated_symbol_stream_distorted_equalized);
ser = serVec(1)
number_of_symbol_errors = serVec(2)








