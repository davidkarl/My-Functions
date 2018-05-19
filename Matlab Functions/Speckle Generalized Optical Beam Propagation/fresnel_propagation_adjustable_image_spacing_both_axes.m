function [mat_out,x2,y2] = fresnel_propagation_adjustable_image_spacing_both_axes(...
                                               A, lambda, source_spacing, image_spacing_x,image_spacing_y, z)
N=size(A,1);
k=2*pi/lambda;

% source-plane coordinates
[x1,y1]=meshgrid((-N/2:N/2-1)*source_spacing);

% magnification of image-spacing from source-spacing
m_x = image_spacing_x/source_spacing;
m_y = image_spacing_y/source_spacing;

% intermediate plane
dz_intermediate_x=z/(1-m_x); %propagation distance
dz_intermediate_y=z/(1-m_y);

intermediate_spacing_x = lambda*abs(dz_intermediate_x)/(N*source_spacing);
intermediate_spacing_y = lambda*abs(dz_intermediate_y)/(N*source_spacing);

[x1a,y1a] = meshgrid((-N/2:N/2-1)*intermediate_spacing_x,(-N/2:N/2-1)*intermediate_spacing_y);

% intermediate fresnel-kirchhoff propagation
mat_intermediate = 1/(1i*lambda*sqrt(dz_intermediate_x)*sqrt(dz_intermediate_y)).* ...
                   exp(1i*k/(2*dz_intermediate_x)*x1a.^2 + 1i*k/(2*dz_intermediate_y)*y1a.^2).* ...
                   ft2(A.*exp(1i*k/(2*dz_intermediate_x)*x1.^2 + 1i*k/(2*dz_intermediate_x)*y1.^2),source_spacing);

% observation-plane coordinates
dz2_x = z-dz_intermediate_x; %propagation distance from intermediate to image
dz2_y = z-dz_intermediate_y;
[x2,y2] = meshgrid((-N/2:N/2-1)*image_spacing_x,(-N/2:N/2-1)*image_spacing_y);

% intermediate to image fresnel-kichhoff integral
mat_out = 1/(1i*lambda*sqrt(dz2_x)*sqrt(dz2_y)).* ...
                exp(1i*k/(2*dz2_x)*x2.^2 + 1i*k/(2*dz2_y)*y2.^2).* ...
                    ft2(mat_intermediate.*exp(1i*k/(2*dz2_x)*x2.^2 + 1i*k/(2*dz2_y)*y2.^2),intermediate_spacing_x,intermediate_spacing_y);





