%test new SHIR idea:

phase_shift_fraction = 200;
phase_shift_every_tic = 2*pi/phase_shift_fraction;
number_of_phase_tics_to_average = phase_shift_fraction;

N = 512;
speckle_size_in_pixels_far_field = 20;
number_of_speckles_on_object = 5;
object_spot_in_pixels = N/speckle_size_in_pixels_far_field;
speckle_size_in_pixels_on_object = object_spot_in_pixels / number_of_speckles_on_object * 2;
x = [-N/2:N/2-1];
[X,Y] = meshgrid(x);
circular_aperture_on_object = (X.^2 + Y.^2 <= object_spot_in_pixels^2);

speckle_pattern_before_object1 = create_speckles_of_certain_size_in_pixels(speckle_size_in_pixels_on_object,N,1,0);
speckle_pattern_before_object2 = create_speckles_of_certain_size_in_pixels(speckle_size_in_pixels_on_object,N,1,0);
surface1 = exp(1i*20*randn(N,N));
surface2 = exp(1i*20*randn(N,N));
 

speckle_pattern_intensity_total_integrated = zeros(N,N);
for phase_counter = 1:number_of_phase_tics_to_average
    speckle_pattern_before_object1 = speckle_pattern_before_object1 .* exp(1i*phase_shift_every_tic);
    
    speckle_pattern_on_object1 = speckle_pattern_before_object1.*surface1.*circular_aperture_on_object;
    speckle_pattern_on_object2 = speckle_pattern_before_object2.*surface1.*circular_aperture_on_object;
    
    speckle_pattern_far_field1 = ft2(speckle_pattern_on_object1,1);
    speckle_pattern_far_field2 = ft2(speckle_pattern_on_object2,1);
    
    speckle_pattern_total = speckle_pattern_far_field1 + speckle_pattern_far_field2;
    speckle_pattern_intensity_total = abs(speckle_pattern_total).^2;
    speckle_pattern_intensity_total_integrated = speckle_pattern_intensity_total_integrated + ...
                                                    speckle_pattern_intensity_total;
end

contrast1 = std(abs(speckle_pattern_far_field1).^2)/mean(abs(speckle_pattern_far_field1).^2);
contrast = std(speckle_pattern_intensity_total_integrated)/mean(speckle_pattern_intensity_total_integrated);
1;
figure
imagesc(abs(speckle_pattern_far_field1).^2);
figure
imagesc(speckle_pattern_intensity_total_integrated);


