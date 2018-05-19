function phase_screen = ft_phase_screen(r0,N,delta,L0,l0)

%L0=outer scale frequency
%l0=inner scale frequency
%delta=real grid spacing
%r0=coherence diameter

% setup the PSD
delta_f=1/(N*delta); %frequency grid spacing [1/meter]
fx=(-N/2:N/2-1)*delta_f;
[fx,fy]=meshgrid(fx);
[theta,f]=cart2pol(fx,fy);

fm=5.92/(2*pi*l0); %inner scale frequency [1/meter]
f0=1/L0; %outer scale frequency [1/meter]

%modified von Karman atmospheric phase PSD
PSD_phi=0.023*r0^(-5/3)*exp(-(f/fm).^2)./(f.^2+f0^2).^(11/6); % i think this is 0.23 but they wrote 0.023!!!!1
PSD_phi(N/2+1,N/2+1)=0; %????

%random draws of Fourier coefficients
cn=((randn(N)+1i*randn(N)).*sqrt(PSD_phi))*delta_f;

%synthesize the phase screen
phase_screen=real(ift2(cn,1));



