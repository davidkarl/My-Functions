function y = apply_positive_negative_modulation(x)
% MirrorFilt -- Apply (-1)^t modulation
%  Usage
%    h = MirrorFilt(l)
%  Inputs
%    l   1-d signal
%  Outputs
%    h   1-d signal with DC frequency content shifted
%        to Nyquist frequency
y = -( (-1).^(1:length(x)) ).*x;