function [sf,frq] = smperio(x,M,window)
%dsp book smoothed periodogram
% x=input sequence
% M such that the window length is 2*M+1
% window = 'r' for rectangular, t for triangular
% frq = frequencies for the estimated PSD
% sf = spectrum

if nargin<3, window='t'; end
x=x(:);
N=length(x);
if nargin<2, M=N^(2/5); end
sf=zers(N-2*M+1,1);
if window=='t'
   Wf=[(1:M+1), (M:-1:1)]/(M+1)^2; 
else
   Wf=ones(1,2*M+1)/(2*M+1);
end
Periodogram = abs(fft(x)).^2/N;
frq=(M+1:N-M-1)/N;
sf=filter(Wf,1,Periodogram);
sf=sf(2*M+2:N);






