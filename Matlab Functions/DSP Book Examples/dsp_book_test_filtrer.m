%dsp book test filterer

inp1=randn(100,1);
inp2=randn(100,1);
etot=[inp1;inp2];
b=[1,0.3];
a=[1,-0.8,0.9];

%global filtering (null initial state);
y=filtrer(b,a,etot);

%filtering the 2 blocks:
[y1,xs]=filtrer(b,a,inp1); %null initial state
y2=filtrer(b,a,inp2,xs); %initial state xs
yp=[y1;y2];

%drawing for the transition between 2 blocks:
[y(90:110), yp(90:110)]



