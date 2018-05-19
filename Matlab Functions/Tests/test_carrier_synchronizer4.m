%test carrier synchronizer4
%This example shows how to model channel impairments such as timing phase offset, carrier frequency offset, and carrier phase offset for a minimum shift keying (MSK) signal. The example also shows the use of System objects™ to synchronize such signals at the receiver.


%Introduction:
%This example models an MSK transmitted signal undergoing channel impairments such as timing, 
%frequency, and phase offset as well as AWGN noise.
%An MSK timing synchronizer recovers the timing offset, 
%while a carrier synchronizer recovers the carrier frequency and phase offsets.

%Initialize system variables by using the MATLAB script configureMSKSignalRecoveryEx. Define logical control variables to enable timing phase and carrier frequency and phase recovery.
configureMSKSignalRecoveryEx;
recoverTimingPhase = true;
recoverCarrier = true;

%Modeling Channel Impairments:
%Specify the sample delay, timingOffset , that the channel model applies, and create a variable fractional delay object to introduce the timing delay to the transmitted signal.
timingOffset = 0.2;
varDelay = dsp.VariableFractionalDelay;

%Introduce carrier phase and frequency offsets by creating a phase and frequency offset object, PFO. Because the MSK modulator upsamples the transmitted symbols, set the SampleRate property appropriately.
freqOffset = 50;
phaseOffset = 30;
pfo = comm.PhaseFrequencyOffset(...
    'FrequencyOffset', freqOffset, ...
    'PhaseOffset', phaseOffset, ...
    'SampleRate', samplesPerSymbol/Ts);

%Create an AWGN channel to add additive white Gaussian noise to the modulated signal. The noise power is determined by the bit energy to noise power spectral density ratio EbNo property. Because the MSK modulator generates symbols with 1 Watt of power, the signal power property of the AWGN channel is also set to 1.
EbNo = 20 + 10*log10(samplesPerSymbol);
chAWGN = comm.AWGNChannel(...
    'NoiseMethod', 'Signal to noise ratio (Eb/No)', ...
    'EbNo', EbNo,...
    'SignalPower', 1, ...
    'SamplesPerSymbol', samplesPerSymbol);

%Timing Phase, Carrier Frequency, and Carrier Phase Synchronization:
%Construct an MSK timing synchronizer to recover symbol timing phase using a fourth-order nonlinearity method.
timeSync = comm.MSKTimingSynchronizer(...
    'SamplesPerSymbol', samplesPerSymbol, ...
    'ErrorUpdateGain', 0.02);

%Construct a carrier syncrhonizer to recover both carrier frequency and phase. Set the modulation to QPSK, because the MSK constellation is QPSK with a 0 degree phase offset.
phaseSync = comm.CarrierSynchronizer(...
    'Modulation', 'QPSK', ...
    'ModulationPhaseOffset', 'Custom', ...
    'CustomPhaseOffset', 0, ...
    'SamplesPerSymbol', 1);

%Stream Processing Loop:
%The system modulates data using MSK modulation. 
%The modulated symbols pass through the channel model, which applies timing delay, 
%carrier frequency and phase shift, and additive white Gaussian noise. 
%In this system, the receiver performs timing phase, and carrier frequency and phase recovery. 
%Finally, the system demodulates the symbols and calculates the bit error rate using an error rate calculator object. 
%The plotResultsMSKSignalRecoveryEx script generates scatter plots to show these effects:
%(1). Channel impairments
%(2). Timing synchronization
%(3). Carrier synchronization
%At the end of the simulation, the example displays the timing phase, frequency, and phase estimates as a function of simulation time.
for p = 1:numFrames
  %------------------------------------------------------------------------
  % Generate and modulate data
  %------------------------------------------------------------------------
  txBits = randi([0 1],samplesPerFrame,1);
  txSym = modem(txBits);
  %------------------------------------------------------------------------
  % Transmit through channel
  %------------------------------------------------------------------------
  %
  % Add timing offset
  rxSigTimingOff = varDelay(txSym,timingOffset*samplesPerSymbol);
  %
  % Add carrier frequency and phase offset
  rxSigCFO = pfo(rxSigTimingOff);
  %
  % Pass the signal through an AWGN channel
  rxSig = chAWGN(rxSigCFO);
  %
  % Save the transmitted signal for plotting
  plot_rx = rxSig;
  %
  %------------------------------------------------------------------------
  % Timing recovery
  %------------------------------------------------------------------------
  if recoverTimingPhase
    % Recover symbol timing phase using fourth-order nonlinearity method
    [rxSym,timEst] = timeSync(rxSig);
    % Calculate the timing delay estimate for each sample
    timEst = timEst(1)/samplesPerSymbol;
  else
    % Do not apply timing recovery and simply downsample the received signal
    rxSym = downsample(rxSig,samplesPerSymbol);
    timEst = 0;
  end

  % Save the timing synchronized received signal for plotting
  plot_rxTimeSync = rxSym;

  %------------------------------------------------------------------------
  % Carrier frequency and phase recovery
  %------------------------------------------------------------------------
  if recoverCarrier
    % The following script applies carrier frequency and phase recovery
    % using a second order PLL, and remove phase ambiguity
    [rxSym,phEst] = phaseSync(rxSym);
    removePhaseAmbiguityMSKSignalRecoveryEx;
    freqShiftEst = mean(diff(phEst)/(Ts*2*pi));
    phEst = mod(mean(phEst),360); % in degrees
  else
    freqShiftEst = 0;
    phEst = 0;
  end

  % Save the phase synchronized received signal for plotting
  plot_rxPhSync = rxSym;
  %------------------------------------------------------------------------
  % Demodulate the received symbols
  %------------------------------------------------------------------------
  rxBits = demod(rxSym);
  %------------------------------------------------------------------------
  % Calculate the bit error rate
  %------------------------------------------------------------------------
  errorStats = BERCalc(txBits,rxBits);
  %------------------------------------------------------------------------
  % Plot results
  %------------------------------------------------------------------------
  plotResultsMSKSignalRecoveryEx;
end



%Display the bit error rate, BitErrorRate , as well as the total number of symbols, NumberOfSymbols , processed by the error rate calculator.

ber = errorStats(1)
numSym = errorStats(3)








