%dsp book test lattice filtering
ai=[1,-1.8,0.9];
N=1000;
wn=randn(N,1);
xn=filter(1,ai,wn);
ki=atok(ai);
[eF,eB]=lattice_analysis(xn,ki);
[xn_s,eB_s]=lattice_synthesis(eF,ki);
[max(abs([wn-eF])) , max(abs([xn-xn_s]))]
figure(1);
plot(xn,'g');
hold on;
plot(xn_s,'r');
hold off;

%forward error is the equivalent of the process noise!!!
figure(2);
plot(eF,'g');
hold on;
plot(wn,'r');
hold off;

figure(3);
plot(eB,'g');
hold on;
plot(eB_s,'r');
hold off;



