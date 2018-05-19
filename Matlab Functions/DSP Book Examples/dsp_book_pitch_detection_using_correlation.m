function [vnv,pitch]=detectpitch(sig,trhld,tmin,tmax,energm)
%dsp pitch detection using correlation:
%sig=signal block
%trhld=correlation threshold
%tmin,tmax=correlation window
%energm=energy threshold
%vnv=true if voiced, otherwise false
%pitch=pitch period
nfa=length(sig);
x=zeros(nfa,1);
x(:)=sig;
ae=x'*x;

if (ae>energm)
    for T=tmin:tmax
       stmT=x(T:nfa);
       s0T=x(1:nfa-T+1);
       autoc=stmT'*s0T;
       etmT=stmT'*stmT;
       e0T=s0T'*s0T;
       correl(T-tmin+1)=autoc/sqrt(etmT*e0T);
    end
    [corrmax,imax]=max(correl);
    tfond=imax+tmin-1;
    if (corrmax<trhld)
       vnv=(0==1);
       pitch=0;
       return;
    else
        pitch=tfond;
        vnv=(0==0);
    end
else
   pitch=-1;
   vnv=(0==0);
end
return;












