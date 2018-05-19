function [mat_out,x2,y2,image_spacing] = fresnel_propagation(A, lambda, spacing, z)
N=size(A,1);
k=2*pi/lambda;

% source-plane coordinates
[x1,y1]=meshgrid((-N/2:N/2-1)*spacing);

% observation-plane coordinates
image_spacing=(1/(N*spacing))*lambda*z;
[x2,y2]=meshgrid((-N/2:N/2-1)*image_spacing);

% fresnel-kichhoff integral
mat_out=1/(1i*lambda*z).*exp(1i*k/(2*z)*(x2.^2+y2.^2)).*ft2(A.*exp(1i*k/(2*z).*(x1.^2+y1.^2)),spacing);








