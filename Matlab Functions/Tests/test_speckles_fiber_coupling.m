%test speckles fiber coupling;

%setup parameters:
z = 1000;
lambda = 1.55*10^-6;
focal_length = 0.3;
mode_field_diameter = 8*10^-6;
receiver_lens_diameter = 0.075;

%imaging parameters:
diameter_on_object_factor = 3;
diameter_on_object = mode_field_diameter * ((z-focal_length)/focal_length) * diameter_on_object_factor;
wf_object = diameter_on_object/2;

%meshgrid parameters:
N = 512*4;
object_view_size = diameter_on_object * 50;
object_spacing = object_view_size/N;
receiver_view_size = lambda*z/object_spacing;
receiver_spacing = receiver_view_size/N;

%create meshgrids:
x_object = [-N/2:N/2-1]*object_spacing;
[X_object,Y_object] = meshgrid(x_object);
x_receiver = [-N/2:N/2-1]*receiver_spacing;
[X_receiver,Y_receiver] = meshgrid(x_receiver);

%create object beam:
gaussian_beam_on_object = exp(-(X_object.^2+Y_object.^2)/(2*wf_object^2));

%Lens without interpolation parameters:
k = 2*pi/lambda;
receiver_lens_radius = receiver_lens_diameter/2;
lens_aperture = (X_receiver.^2+Y_receiver.^2<=receiver_lens_radius.^2);
Lens = lens_aperture.*exp(-1i*k/(2*focal_length)*(X_receiver.^2+Y_receiver.^2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%scatter gaussian beam and create speckles:
[speckle_pattern] = fresnel_propagation(gaussian_beam_on_object.*exp(1i*20*randn(N,N)),lambda,object_spacing,z);

%interpolate speckle pattern so that whole view size will be lens:
% speckle_pattern = interp2(X_receiver,Y_receiver,speckle_pattern,X_lens,Y_lens);

%pass through lens:
speckle_pattern_after_lens = speckle_pattern.*Lens;
% speckle_pattern_after_lens = speckle_pattern.*Lens_whole_view;

%propagate to fiber:
image_distance = (1/focal_length-1/z)^-1;
image_spacing = lambda*focal_length/receiver_view_size;
image_view_size = image_spacing*N;
speckle_pattern_on_fiber = angular_spectrum_propagation(speckle_pattern_after_lens,lambda,receiver_spacing,image_spacing,image_distance);
 
%find coupling efficiency with fiber:
x_fiber = [-N/2:N/2-1]/N*image_view_size;
[X_fiber,Y_fiber] = meshgrid(x_fiber);
w0_fiber = mode_field_diameter/2;
fiber_gaussian = exp(-(X_fiber.^2+Y_fiber.^2)/(2*w0_fiber^2));
coupling_nominator = abs(sum(sum(speckle_pattern_on_fiber.*fiber_gaussian))).^2;
coupling_denominator = sum(sum(abs(speckle_pattern_on_fiber).^2)) .* sum(sum(abs(fiber_gaussian).^2));
coupling_efficiency = coupling_nominator/coupling_denominator;
1;











