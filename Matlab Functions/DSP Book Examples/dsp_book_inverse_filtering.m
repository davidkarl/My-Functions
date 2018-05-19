%dsp book inverse filtering:
filter_psf=[1,-3.4,1.2];
N=length(filter_psf);

%causal and anticausal length:
C=20;
A=10;
%impulse 0...0 1 0..0
delta=zeros(N+A+C,1);
delta(A+1)=1;
filter_PSD_toeplitz_matrix = toeplitz([filter_psf,zeros(1,A+C)],[filter_psf(1),zeros(1,A+C)]);

% conv(PSF,g) == PSF_toeplitz_matrix * g_input_signal = delta_output_signal:
g = filter_PSD_toeplitz_matrix\delta;

%verification using a convolution:
stem(conv(g,filter_psf))

%verification using the theoretical values:
gA=-(3.^(0:-1:-A+2))/7.8;
gC=-(0.4.^(0:C+1))/2.6;
gth=[gA(A-1:-1:1),gC]';
[gth,g],
plot(g);
hold on;
plot(gth,'r') 
max(abs(gth-g))



















