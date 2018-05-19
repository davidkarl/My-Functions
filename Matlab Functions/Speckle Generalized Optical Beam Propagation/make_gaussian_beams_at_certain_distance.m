function [gaussian_beam]=make_gaussian_beams_at_certain_distance(lambda,wf,distance_x,distance_y,spacing,N,z)
x=[-N/2:N/2-1]*spacing;
[X,Y]=meshgrid(x);

w0 = sqrt(max(roots([1,-wf^2,(lambda*z/pi)^2]))); %using paraxial gaussian beam equations
Zr=pi*w0^2/lambda; % Rayleigh range
R=z*(1+(Zr/z)^2); %curvature radius
k=2*pi/lambda;
zeta=atan(z/Zr);

%first beam
gaussian_beam=exp(-((X-distance_x/2).^2+(Y-distance_y/2).^2)/wf^2).*exp(-1i*k/(2*R)*((X-distance_x/2).^2+(Y-distance_y).^2));
%second beam
gaussian_beam=gaussian_beam + exp(-((X+distance_x/2).^2+(Y+distance_y/2).^2)/wf^2).*exp(-1i*k/(2*R)*((X+distance_x/2).^2+(Y+distance_y/2).^2));
%global phase
gaussian_beam=gaussian_beam*exp(-1i*k*z)*exp(+1i*zeta);
%normalize
gaussian_beam=gaussian_beam/sqrt(sum(sum(abs(gaussian_beam).^2))); 

