trueCoef=[1,-1.3,0.8];
P=length(trueCoef)-1;
sw=sqrt(1);
Lrun=100;
listN=(500:500:4000);
lgN=length(listN);
perf=zeros(lgN,1);

for ii=1:lgN
   N=listN(ii);
    for ell=1:Lrun
        w=sw*randn(N,1);
        x=filter(1,trueCoef,w);
        [aest,s2est]=xtoa(x,P);
        
        %performance for the estimation of the coeff trueCoef(2)
        eQ=(aest(2)-trueCoef(2))*(aest(2)-trueCoef(2))';
        perf(ii)=perf(ii)+eQ;
    end
end
perf=perf/Lrun;
plot(listN,perf);
hold;
plot(listN,perf,'ro');
hold;
grid;







