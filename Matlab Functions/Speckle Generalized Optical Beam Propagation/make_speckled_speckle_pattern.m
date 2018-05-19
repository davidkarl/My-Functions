function [far_field_speckle_pattern,incident_speckle_pattern] = make_speckled_speckle_pattern(speckle_size_corresponding_to_beam_size, incident_speckle_size_to_beam_diameter_ratio, N)
% speckle_size_corresponding_to_beam_size = 10;
% incident_speckle_size_to_beam_diameter_ratio = 0.01;
% N=1024/2;

%nominal beam radius [pixels]:
wf = N/speckle_size_corresponding_to_beam_size; 

%create incident truncated speckle pattern:
incident_speckle_size_in_pixels = wf*incident_speckle_size_to_beam_diameter_ratio;
incident_speckle_pattern = create_speckles_of_certain_size_in_pixels(incident_speckle_size_in_pixels,N,1,0);
circular_truncation_aperture = make_circular_beams_profile(wf,0,0,N,1);
incident_speckle_pattern = incident_speckle_pattern.*circular_truncation_aperture;

%create far field speckle pattern:
incident_speckle_pattern = incident_speckle_pattern.*exp(1i*10*randn(N,N));
far_field_speckle_pattern = ft2(incident_speckle_pattern,1);

% a=imhist(abs(far_field_speckle_pattern).^2,100);
% figure(1)
% imagesc(abs(incident_speckle_pattern));
% figure(2)
% imagesc(abs(far_field_speckle_pattern));

