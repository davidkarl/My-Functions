function [speckle_fringes] = make_speckle_fringes_in_the_far_field(beam_size,distance_x,distance_y,z,lambda,far_field_view_size,N,gaussian_beam_flag)

%CREATE THE ROUGH SURFACE:
[rough_surface,surface_spacing,X_surface,Y_surface]=create_rough_surface_with_proper_size_and_spacing(lambda,10,z,far_field_view_size,N);
   
%CREATE SPECKLED BEAMS ON SURFACE:
if gaussian_beam_flag==1
    [total_beam,beam_one,beam_two,X,Y]=make_gaussian_beams_profile(beam_size,distance_x,distance_y,N,surface_spacing);
else
    [total_beam,beam_one,beam_two,X,Y]=make_circular_beams_profile(beam_size,distance_y,distance_y,surface_spacing,N);
end

two_beams_on_surface = beam_one + beam_two;
two_beams_on_surface = two_beams_on_surface.*rough_surface;

speckle_fringes = fresnel_propagation(two_beams_on_surface,lambda,surface_spacing,z);





