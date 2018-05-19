
x = randn(2048,5);
x = x-mean(x);
src = dsp.SignalSource;
src.Signal = x;
sink = dsp.SignalSink;

N = 10;
Fc = 0.4;
[b,a] = butter(N,Fc);
iir = dsp.IIRFilter('Numerator',b,'Denominator',a);
% iir.freqz

% bla = step(iir,x);
% figure;
% bli = 10*log10(abs(fftshift(fft(bla(:,2)))));
% plot(bli(end/2+1:end));

sa = dsp.SpectrumAnalyzer('SampleRate',8e3,...
    'PlotAsTwoSidedSpectrum',false,...
    'OverlapPercent',80,'PowerUnits','dBW',...
    'YLimits',[-220 -10]);

counter = 1;
while ~isDone(src) 
  input = src();
  output = iir(input);
  sa(output)
  sink(output);
  counter = counter + 1;
  counter
end

% Result = sink.Buffer;
% fvtool(iir,'Fs',8000)



