function [mat_out,x2,y2] = fresnel_propagation_adjustable_image_spacing(A, lambda, source_spacing, image_spacing, z)
N=size(A,1);
k=2*pi/lambda;

% source-plane coordinates
[x1,y1]=meshgrid((-N/2:N/2-1)*source_spacing);

% magnification of image-spacing from source-spacing
m=image_spacing/source_spacing;

% intermediate plane
dz_intermediate=z/(1-m); %propagation distance
intermediate_spacing=lambda*abs(dz_intermediate)/(N*source_spacing);
[x1a,y1a]=meshgrid((-N/2:N/2-1)*intermediate_spacing);
% intermediate fresnel-kirchhoff propagation
mat_intermediate=1/(1i*lambda*dz_intermediate).*exp(1i*k/(2*dz_intermediate)*(x1a.^2+y1a.^2)).*ft2(A.*exp(1i*k/(2*dz_intermediate)*(x1.^2+y1.^2)),source_spacing);

% observation-plane coordinates
dz2=z-dz_intermediate; %propagation distance from intermediate to image
[x2,y2]=meshgrid((-N/2:N/2-1)*image_spacing);
% intermediate to image fresnel-kichhoff integral
mat_out=1/(1i*lambda*dz2).*exp(1i*k/(2*dz2)*(x2.^2+y2.^2)).*ft2(mat_intermediate.*exp(1i*k/(2*dz2)*(x1a.^2+y1a.^2)),intermediate_spacing);





