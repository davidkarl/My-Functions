function ai=ktoa(ki)
%dsp book convert reflection coefficients to AR parameters
%ki =reflection coefficients (k1,...kp)
%ai = AR model parameters (1,a1,...ap)
P=length(ki);
ai=[1;zeros(P,1)];
for ii=1:P
   ai(1:ii+1)=ai(1:ii+1)+ki(ii)*conj(ai(ii+1:-1:1)); 
end






