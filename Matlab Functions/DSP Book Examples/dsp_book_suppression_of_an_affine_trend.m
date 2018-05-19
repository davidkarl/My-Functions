function [A,dx]=tendoff(x)
%dsp book suppression of an affine trend
%x=input sequence
%A=Affine regression coeffts
%dx=residual
x=x(:);
N=length(x);
w=[ones(N,1),(0:N-1)'];
A=w\x;
dx=x-w*A;
return



