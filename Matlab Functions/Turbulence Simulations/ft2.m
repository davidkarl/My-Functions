function G = ft2(g, spacing, final_rows, final_cols, varargin)
if nargin==2
    G = fftshift(fft2(fftshift(g))) * spacing^2;
elseif nargin==3
    G = fftshift(fft2(fftshift(g),final_rows,final_rows)) * spacing^2;
elseif nargin==4
    G = fftshift(fft2(fftshift(g),final_rows,final_cols)) * spacing^2;
end

% spacing=L[meters]/N[number of samples]
% f_spacing=1/(N*spacing)=1/L[meters]
% x=(-N/2:N/2-1)*spacing
% fx=(-1/2:1/2-1/N)/spacing;



 