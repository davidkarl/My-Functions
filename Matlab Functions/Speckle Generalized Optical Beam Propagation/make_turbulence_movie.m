%record beam on surface and speckle pattern movies in the presence of turbulence:



clear all;
clc;

N=512;  
l0_vec = linspace(15,15,1)*10^-3;
L0_vec = linspace(50,50,1);
lambda = 1.55*10^-6;
z_initial = 0;
z_vec = linspace(10000,10000,1);
wf_vec = linspace(10,10,1)*10^-2;
distance_between_beams = 0;
height_std=20;
c=3*10^8;
wind_velocity_variance = 5;
minimum_number_of_screens=20; %minimum number of phase screens to get some continuity to the discrete simulation
Cn2_vec=logspace(-15,-15,1);

number_of_different_realizations = 1;
simulation_number_of_steps = 200;

for l0_counter=1:length(l0_vec)
    for L0_counter=1:length(L0_vec)
        for z_counter=1:length(z_vec)
            for wf_counter=1:length(wf_vec)
                for Cn2_counter=1:length(Cn2_vec)
                    
                    for s=1:number_of_different_realizations
                        
                        %BUILD SCATTERIG SURFACE:
                        surface = exp(1i*20*randn(N,N));
                        
                        %TURBULENCE PARAMETERS:
                        l0=l0_vec(l0_counter);
                        L0=L0_vec(L0_counter);
                        z=z_vec(z_counter);
                        wf=wf_vec(wf_counter);
                        w0 = sqrt(max(roots([1,-wf^2,(lambda*z/pi)^2]))); %using paraxial gaussian beam equations
                        Cn2=Cn2_vec(Cn2_counter);  
                        
                        %BUILD MOVIE STRINGS:
                        surface_movie_string = strcat('long distance3 surface beam with l0=',num2str(l0),',L0=',num2str(L0),',Cn2=',num2str(Cn2),',z=',num2str(z),',wf=',num2str(wf));
                        speckle_movie_string = strcat('long distance3 with l0=',num2str(l0),',L0=',num2str(L0),',Cn2=',num2str(Cn2),',z=',num2str(z),',wf=',num2str(wf));
                        surface_beam_movie_writer = VideoWriter(strcat(surface_movie_string,'.avi'));
                        speckle_beam_movie_writer = VideoWriter(strcat(speckle_movie_string,'.avi'));
                        open(surface_beam_movie_writer);
                        open(speckle_beam_movie_writer);
                        
                        %FORWARD PROPAGATION SPACINGS:
                        initial_view_size = 1;
                        surface_view_size = 1;
                        surface_spacing = surface_view_size/N;
                        delta1 = initial_view_size/N;
                        delta2 = surface_view_size/N;                       
                        
                        %SCREEN AND TURBULENCE "MACRO" PARAMETERS:
                        [rytov,r0sw,r0pw,t_coherence,r0_screens,number_of_screens,d_r0sw,d_rytov] = calculate_turbulence_parameters(Cn2,z,lambda,wind_velocity_variance,minimum_number_of_screens);
                        
                        %MAKE LONGITUDINAL PROPAGATION PARAMETERS:
                        z_vec1=linspace(0,z,number_of_screens);
                        delta_z=z_vec1(2)-z_vec1(1);
                        alpha1=z_vec1/z;
                        delta_with_distance = (1-alpha1)*delta1+alpha1*delta2;
                        
                        %TYPICAL TIMES:
                        total_number_of_coherence_times_to_simulation = 20;
                        simulation_total_time = total_number_of_coherence_times_to_simulation/0.31*t_coherence;
                        simulation_time_step = simulation_total_time/simulation_number_of_steps; %presumably the shorter the better
                        t_partial_propagation = delta_z/c;
                        t_full_propagation = z/c;
                        %i rely on the fact that usually v_wind*t_propagation << simulation_view_size

                        %MAKE INITIAL GAUSSIAN BEAMS:
                        [total_beam,beam_one,beam_two,X,Y]=make_gaussian_beams_profile(w0,0,0,N,delta1);
                        mat_in_with_turbulence = total_beam;
                        
                        %FINAL SURFACE INTERPOLATION COORDINATES:
                        final_far_field_desired_view_size = 1;
                        final_far_field_spacing = final_far_field_desired_view_size/N;
                        final_surface_view_size = lambda*z/final_far_field_spacing;
                        final_surface_spacing = final_surface_view_size/N;
                        x_final_surface = [-N/2:N/2-1]*final_surface_spacing;
                        [X_final_surface,Y_final_surface] = meshgrid(x_final_surface);
                        
                        %FINAL FAR FIELD SPECKLE COORDINATES:
                        %set first propagation screen size and spacing:
                        far_field_speckle_view_size = lambda*z/final_surface_spacing;
                        far_field_speckle_spacing = far_field_speckle_view_size/N;
                        view_size_for_first_propagation_from_surface = far_field_speckle_view_size * (delta_z/z);
                        first_propagation_far_field_spacing = view_size_for_first_propagation_from_surface/N;
                        %define propagation parameters for backward propagation:
                        D1=0;
                        delta1=0;
                        D2=far_field_speckle_view_size;
                        delta2=D2/N;
                        z_vec2=linspace(0,z,number_of_screens);
                        delta_z2=z_vec2(2)-z_vec2(1);
                        alpha2=z_vec2/z;
                        delta_with_distance2=(1-alpha2)*delta1+alpha2*delta2;
                        
                        %INTERPOLATION COORDINATES:
                        X_interpolates = zeros(N,N,number_of_screens-1);
                        Y_interpolates = zeros(N,N,number_of_screens-1);
                        X_originals = zeros(N,N,number_of_screens-1);
                        Y_originals = zeros(N,N,number_of_screens-1);
                        for t=2:number_of_screens
                            x_interpolate=[N/2-1:-1:-N/2]*delta_with_distance2(t);
                            [X_interpolates(:,:,t),Y_interpolates(:,:,t)]=meshgrid(x_interpolate); 
                            x_original=[-N/2:N/2-1]*delta_with_distance(t);
                            [X_originals(:,:,t),Y_originals(:,:,t)]=meshgrid(x_original);
                        end
                        Y_interpolates = -Y_interpolates; %the only flip is in the x-direction
                        
                        
                        %BUILD FORWARD PROPAGATION PHASE SCREENS:
                        %i try to balance efficiency, memory, and the need for the same phase screens 
                        %front and back. i resort to interpolation:
                        number_of_propagations = number_of_screens - 1;
                        subharmonics_accuracy = 0.99;
                        L_simulation = min(initial_view_size,view_size_for_first_propagation_from_surface);
                        number_of_subharmonics_forward = ceil((1/log(3))*log( (L0/L_simulation)*(subharmonics_accuracy^(-5/6)-1)^(-1/2) )); 
                        wind_velocity_magnitudes = repmat(wind_velocity_variance*randn(number_of_screens,1),1,2);
                        wind_velocity_phases = 2*pi*rand(number_of_screens,1);
                        wind_velocity_directions = [cos(wind_velocity_phases),sin(wind_velocity_phases)];
                        wind_velocities = wind_velocity_magnitudes .* wind_velocity_directions;
                        
                        cn_high_forward = zeros(N,N,number_of_screens);
                        cn_low_forward = zeros(3,3,number_of_subharmonics_forward,number_of_screens);
                        for t=1:number_of_screens
                           [cn_high_forward(:,:,t),cn_low_forward(:,:,:,t)] = get_turbulence_screen_fourier_coefficients(r0_screens(t),N,delta_with_distance(t),L0,l0,number_of_subharmonics_forward);
                        end 
                        forward_phase_screens = zeros(N,N,number_of_screens);
                       
                       [expansion_phase] = make_phase_for_proper_beam_expansion(wf/w0,z,lambda,delta_with_distance(1),N,0,0);
                       current_time = 0;
                        
                        %ACTUALLY CARRY OUT THE TEMPORAL SIMULATION: 
                        for time_step=1:simulation_number_of_steps
                            tic
                            %ASSIGN INITIAL BEAMS:
                            mat_in_with_turbulence = total_beam.*expansion_phase;
                            
                            %PROPAGATE THROUGH TURBULENCE FORWARD DIRECTION:
                            for t=1:number_of_propagations+1
                                shiftx = wind_velocities(t,1)*current_time;
                                shifty = wind_velocities(t,2)*current_time;
                                [forward_phase_screens(:,:,t)] = realize_shifted_phase_screen(squeeze(cn_high_forward(:,:,t)),squeeze(cn_low_forward(:,:,:,t)),N,number_of_subharmonics_forward,shiftx,shifty,delta_with_distance(t)); 
                                
                                mat_in_with_turbulence = mat_in_with_turbulence.*exp(1i*squeeze(forward_phase_screens(:,:,t)));
                                  
                                if ~(t==number_of_propagations+1)
                                [mat_in_with_turbulence,X_propagation,Y_propagation]=angular_spectrum_propagation(mat_in_with_turbulence,lambda,delta_with_distance(t),delta_with_distance(t+1),delta_z);  
                                end
                                
                            end  
                           
 
                            
                            %CAPTURE FRAME OF TURBULENT BEAM ON SCREEN:
                            figure(1)
                            imagesc(abs(mat_in_with_turbulence).^2);
                            F_surface = getframe;
                            writeVideo(surface_beam_movie_writer,F_surface);
                            
                            
%                             %INTERPOLATE BEAM ON SURFACE OF PROPER SIZE FOR FOURIER STYLE PROAPGATION:
%                             mat_in_with_turbulence = interp2(X_propagation,Y_propagation,mat_in_with_turbulence,X_final_surface,Y_final_surface,'spline');
%                             mat_in_with_turbulence(isnan(mat_in_with_turbulence))=0;
%                             %BUILD RANDOM SURFACE AND SCATTER BEAMS:
%                             mat_in_with_turbulence = mat_in_with_turbulence.*surface;
%                             
% 
%                             %FRESNEL PROPAGATE TO FIRST BACKWARD PHASE SCREEN:
%                             if wf^4/(8*delta_z^3)<lambda/10 %fresnel diffraction condition
%                             [speckle_pattern_with_forward_turbulence_with_back_turbulence,x2,y2,image_spacing] = fresnel_propagation(mat_in_with_turbulence, lambda, final_surface_spacing, delta_z2);
%                             else 
%                             [speckle_pattern_with_forward_turbulence_with_back_turbulence,x2,y2]=angular_spectrum_propagation(mat_in_with_turbulence,lambda,final_surface_spacing,lambda*delta_z/(N*final_surface_spacing),delta_z2); 
%                             end 
% %                             figure(1)
% %                             imagesc(abs(speckle_pattern_with_forward_turbulence_with_back_turbulence));
% 
% 
%                             %BACK PROPAGATE SPECKLE PATTERNS THROUGH TURBULENCE:
%                             for t=2:number_of_screens
% 
%                                 %interpolate backward phase screen:                               
%                                 shifted_phase_screen = interp2(X_originals(:,:,t),Y_originals(:,:,t),squeeze(forward_phase_screens(:,:,end+1-t)),X_interpolates(:,:,t),Y_interpolates(:,:,t),'spline');
%                                 shifted_phase_screen(isnan(shifted_phase_screen))=0;
%                                 
%                                 %back propagate FORWARD TURBULENCE BACK TURBULENCE BEAM:              
%                                 speckle_pattern_with_forward_turbulence_with_back_turbulence = speckle_pattern_with_forward_turbulence_with_back_turbulence.*exp(1i*shifted_phase_screen);    
%                                 
%                                 if ~(t==number_of_screens)
%                                 [speckle_pattern_with_forward_turbulence_with_back_turbulence,x2,y2]=angular_spectrum_propagation(speckle_pattern_with_forward_turbulence_with_back_turbulence,lambda,delta_with_distance2(t),delta_with_distance2(t+1),delta_z2);
%                                 end
%                                 
% %                                 figure(3)
% %                                 imagesc(abs(speckle_pattern_with_forward_turbulence_with_back_turbulence));
%                             end
%                         
%                             %CAPTURE FRAME OF TURBULENT SPECKLES ON LENS:
%                             figure(2)
%                             imagesc(abs(speckle_pattern_with_forward_turbulence_with_back_turbulence).^2);
%                             F_speckles = getframe;
%                             writeVideo(speckle_beam_movie_writer,F_speckles);
%                             
                             
                            toc 
                            current_time = simulation_time_step*time_step;
                        end %end of current time loop
                      
                        close(surface_beam_movie_writer);
                        close(speckle_beam_movie_writer);
                    end %end of different realizations (for current parameters) loop
                    
                end 
            end 
        end 
    end
end
       
















