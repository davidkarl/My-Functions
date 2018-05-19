function ki=atok(ai)
%dsp book convert AR parameters to reflection coefficients 
%ai=AR model [1,a1,...ap]
%ki=reflection coefficients (k1,...kp)
P=length(ai)-1;
ki=zeros(P,1);
for ii=P:-1:1
   ki(ii)=ai(ii+1);
   bi=conj(ai(ii+1:-1:1));
   umodk2=1-ki(ii)*ki(ii)';
   ai=(ai-ki(ii)*bi)/umodk2;
   ai(ii+1)=[];
end






