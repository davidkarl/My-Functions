function [phase_low,phase_high] = ft_sh_phase_screen_tunable_accuracy(r0,N,delta,L0,l0,Np)

%first of al, we'll get the initial phase screen, consistent with theory
%for high frequencies, from the already built ft_phase_screen function:
phase_high=ft_phase_screen(r0,N,delta,L0,l0);

D=N*delta;
[x,y]=meshgrid((-N/2:N/2-1)*delta);

%initialize low frequency screen
phase_low=zeros(size(phase_high));

%loop over frequency grids with spacing 1/(3^p*L);
for p=1:Np
    % setup the PSD
    delta_f=1/(3^p*D);
    fx=(-1:1)*delta_f;
    [fx,fy]=meshgrid(fx);
    [theta,f]=cart2pol(fx,fy);
    fm=5.92/(2*pi*l0);
    f0=1/L0;
    PSD_phi=0.023*r0^(-5/3)*exp(-(f/fm).^2)./(f.^2+f0^2).^(11/6);
    PSD_phi(2,2)=0; %????
    
    cn=(randn(3)+1i*randn(3)).*sqrt(PSD_phi)*delta_f;
    SH=zeros(N);
    
    %loop over frequencies of this grid
    for ii=1:9
        SH=SH+cn(ii)*exp(2*pi*1i*(fx(ii)*x+fy(ii)*y));
    end
    phase_low=phase_low+SH; %accumulate subharmonics
end

phase_low=real(phase_low)-mean(real(phase_low(:)));

%the complete phase screen is:
%  phase_total=phase_low+phase_high;





