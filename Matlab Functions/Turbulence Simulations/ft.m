function G = ft(g, spacing)
G = fftshift(fft(g)) * spacing;

% spacing=L[meters]/N[number of samples]
% f_spacing=1/(N*spacing)=1/L[meters]
% x=(-N/2:N/2-1)*spacing
% fx=(-1/2:1/2-1/N)/spacing;
