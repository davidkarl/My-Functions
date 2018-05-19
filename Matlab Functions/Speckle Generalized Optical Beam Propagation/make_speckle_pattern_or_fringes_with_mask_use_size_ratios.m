function [speckle_pattern,total_beam,mask,modulation_signal,modulation,max_difference,one_dimensional_speckle_pattern,raw_modulation,total_energy_relative_to_expected_mean] = make_speckle_pattern_or_fringes_with_mask_use_size_ratios(...
    N,...
    speckle_to_fin_size_ratio,...
    fringe_to_fin_size_ratio,... %this should be changed to fringe_to_fin_pair_size_ratio modulation is maximum when fringe_to_fin_size_ratio is 1
    flag_y_separation,...
    number_of_fin_pairs,...
    Fs_to_carrier_ratio_for_modulation_calculation,...
    flag_speckle_blender,...
    flag_calculate_modulation_signal,...
    flag_move_speckle_pattern_or_mask,... %insert 1 to move speckle pattern and 0 otherwise
    flag_calculate_fft_of_signal_modulation,...
    flag_change_intensity_in_speckle_blender,...
    flag_change_phase_in_speckle_blender,...
    speckle_blender_number_of_samples,...
    flag_draw_speckle_blender_pattern_getting_built,...
    flag_draw_surface_and_far_field_graphs,...
    flag_draw_far_field_with_mask_graph,...
    flag_draw_maskless_modulation_signal,...
    flag_draw_maskless_modulation_signal_with_DC,...
    flag_draw_maskless_modulation_fft,...
    flag_draw_modulation_signal,...
    flag_draw_fft_of_signal_modulation,...
    number_of_samples_per_integration_period,...
    number_of_periods)
 
    

% %example of variables to input into function:
% N=1024;
% speckle_to_fin_size_ratio=15; %put inf for tiny beam size
% fringe_to_fin_size_ratio=2; %put inf for no beam separation (one beam BOHA)
% flag_y_separation=1;
% number_of_fin_pairs=25;
% Fs_to_carrier_ratio_for_modulation_calculation=3;
% flag_speckle_blender = 0;
% flag_calculate_modulation_signal=1;
% flag_change_intensity_in_speckle_blender = 1;
% flag_change_phase_in_speckle_blender = 1;
% speckle_blender_number_of_samples=10;
% flag_draw_speckle_blender_pattern_getting_built=0; 
% flag_draw_surface_and_far_field_graphs=1;
% flag_draw_far_field_with_mask_graph=1;
% flag_draw_maskless_modulation_signal=1;
% flag_draw_maskless_modulation_signal_with_DC=1;
% flag_draw_maskless_modulation_fft=0;
% flag_draw_modulation_signal=1;
% flag_draw_fft_of_signal_modulation = 0;
% number_of_samples_per_integration_period=2;
% number_of_periods=2;




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%IF THE NUMBER OF FINS WANTED, OR SPECKLE_TO_FIN_SIZE_RATIO WANTED DOESN'T
%ALLOW US TO FAITHFULLY REPLICATE RAYLEIGH STATISTICS (SEPARATION TOO LARGE
%OR NUMBER OF SPECKLES<8) THAN WE CHANGE THE SEPARATION BETWEEN THE BEAMS
%AND IF THAT'S NOT ENOUGH WE ENLARGE THE N(!!!).

%PSEUDO-ALGORITHM IS:
% (1). put beam separation and beam size with exact number of fringes, which means
%      minimum separation, minimum beam size, and minimum number of speckles.
% (2). check if there are more than 6 or 7 speckles (enough for rayleigh statistics)
% (3). if there are more than 6 or 7 speckles than all is okay
% (4). if there are less than 6 or 7 speckles to multiply beam size and beam separation
%      both by the same amount in order allow more speckles into the view, THIS WOULD
%      REQUIRE LATER SPLINE INTERPOLATION(!!!!!!!).
% (5). if the required separation after step (4) makes number_of_pixels_per_fringe smaller
%      than some minimum, let's say 8 or 9, than multiply N and make it bigger by the 
%      minimum ratio in order to be able to have at least 8 or 9 pixels per fringe meaning
%      that: number_of_fringes = total_separation_in_pixels;
%            number_of_pixels_per_fringe = N/number_of_fringes;
%            -->
%            number_of_pixels_per_fringe = N/total_separation_in_pixels;
           
  
%INITIAL SPACING AND SIZES:
far_field_view_size = N; %simply by definition
far_field_spacing = far_field_view_size/N;
surface_view_size = N/far_field_spacing;
surface_spacing = surface_view_size/N;
%SIZES IN [PIXELS]:
%WITH THIS DEFINITION:
% (1). fringe_to_fin_size_ratio=1 means a fringe size (node to node) is 1 fin size.
%                                 this also means 2*number_of_fin_pairs=number_of_fringes.
%                                 this also mean beam_separation = 2*number_of_fin_pairs [pixels]
% (2). speckle_to_fin_size_ratio=1 means a speckle size is 1 fin size.
%                                  this also means 2*number_of_fin_pairs=number_of_speckles.
fin_size = far_field_view_size/(2*number_of_fin_pairs);
fringe_size = fin_size*fringe_to_fin_size_ratio;
speckle_size = fin_size*speckle_to_fin_size_ratio;
total_beam_separation = (surface_view_size/fringe_size);
beam_radius = (surface_view_size/speckle_size)/2;
original_total_beam_separation = total_beam_separation;
original_beam_radius = beam_radius;

%START WHILE LOOP UNTIL ALL CONDITIONS ARE FULLFILLED:
flag_conditions = 1;
N_ratio_multiplication_factor = 1;
size_ratio_multiplication_factor = 1;
maximum_N_for_simulation = 2048;
N_original = N;
while flag_conditions==1
    flag_conditions = 0;
    
    %make sure there is enough space for enough fringes + beam radiuses:
    if total_beam_separation+beam_radius>N
       N_ratio_multiplication_factor = N_ratio_multiplication_factor*1.5*((total_beam_separation+beam_radius)/N);
       N=ceil(N_ratio_multiplication_factor*N_original);
       far_field_view_size = N; %simply by definition
       far_field_spacing = far_field_view_size/N;
       surface_view_size = N/far_field_spacing;
       surface_spacing = surface_view_size/N;

       %SIZES IN [PIXELS]:
       fin_size = far_field_view_size/(2*number_of_fin_pairs);
       fringe_size = fin_size*fringe_to_fin_size_ratio;
       speckle_size = fin_size*speckle_to_fin_size_ratio;
       total_beam_separation = (surface_view_size/fringe_size);
       beam_radius = (surface_view_size/speckle_size)/2;
    end

    %make sure there are enough speckles in the frame for rayleigh statistics:
    number_of_speckles_in_the_image = 2*beam_radius;
    if number_of_speckles_in_the_image<8 
       size_ratio_multiplication_factor = size_ratio_multiplication_factor*ceil(8/number_of_speckles_in_the_image);
       total_beam_separation = original_total_beam_separation*size_ratio_multiplication_factor;
       beam_radius = original_beam_radius*size_ratio_multiplication_factor;
       number_of_speckles_in_the_image = 2*beam_radius;
    end 
 
    %make sure there are enough pixels per fringe
    number_of_fringes = total_beam_separation;
    number_of_pixels_per_fringe = N/number_of_fringes;
    %-->
    number_of_pixels_per_fringe = N/total_beam_separation;

    if number_of_pixels_per_fringe<8
       N_ratio_multiplication_factor = N_ratio_multiplication_factor*8/number_of_pixels_per_fringe; 
       N = N_original*N_ratio_multiplication_factor;
       number_of_pixels_per_fringe = N/total_beam_separation;
    end
    
    %check all conditions:
    if number_of_pixels_per_fringe<8 || number_of_speckles_in_the_image<6 || total_beam_separation+beam_radius>N
       flag_conditions = 1;
    end
    
    %check N is not too large for my computer to handle:
    if N>maximum_N_for_simulation
       error('N too big for this computer');
    end
end

%ASSIGN number_of_fringes TO KNOW BY HOW MUCH TO ZOOM AND SPLINE INTERPOLATE LATER:
number_of_fringes = total_beam_separation;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ASSIGN CONSTANTS AND BEAM DIRECTION:
%constants:
number_of_samples_for_fft = 5000;
graph_number = 1;
 
if flag_y_separation==1
   distance_x=0;
   distance_y=total_beam_separation;
else 
    distance_x=total_beam_separation; 
    distance_y=0; 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MAKE SPECKLE PATTERN:
rough_surface = exp(1i*10*pi*randn(N,N));
if flag_speckle_blender==1
   speckle_pattern=zeros(N,N);
   for k=1:speckle_blender_number_of_samples
%        [total_beam,beam_one,beam_two,~,~]=make_gaussian_beams_profile(beam_radius,distance_x,distance_y,N,surface_spacing);
       [total_beam,beam_one,beam_two,~,~]=make_circular_beams_profile(beam_radius,distance_x,distance_y,surface_spacing,N);
       beam_speckles = ones(N,N);
       if flag_change_intensity_in_speckle_blender==1
          beam_speckles = beam_speckles.*randn(N,N);
       end
       if flag_change_phase_in_speckle_blender==1
          beam_speckles = beam_speckles.*exp(1i*10*pi*randn(N,N)); 
       end
       beam_one = beam_speckles.*beam_one; 
       beam_two = beam_speckles.*beam_two;
       beam_one = beam_one/sqrt(sum(sum(abs(beam_one).^2)));
       beam_two = beam_two/sqrt(sum(sum(abs(beam_two).^2)));
       total_beam = beam_one+beam_two;
       total_beam = total_beam.*rough_surface;
       speckle_pattern = speckle_pattern + abs(ft2(total_beam,surface_spacing)).^2;
       if flag_draw_speckle_blender_pattern_getting_built==1
           figure(graph_number)
           imagesc(abs(speckle_pattern));
           pause(0.1);
           graph_number=graph_number+1;
       end
   end
   str = strcat('number of different speckle blender patterns = ',num2str(speckle_blender_number_of_samples));
else %ELSE- NO SPECKLE BLENDER, REGULAR BOHA:
   %I ASSUME I DON'T SIMULATE SPECKLES SMALER THAN ONE
   if number_of_speckles_in_the_image<2*N %THIS IS NOT GOOD, I NEED TO FIRST EXPAND THEN ZOOM IN
      [total_beam,beam_one,beam_two,X,Y]=make_gaussian_beams_profile(beam_radius,distance_x,distance_y,N,surface_spacing);
   else
      total_beam = ones(N,N); 
   end
   total_beam = total_beam.*rough_surface;
   speckle_pattern = ft2(total_beam,surface_spacing);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SPLINE INTERPOLATE IF NECESSARY:
if N>N_original || size_ratio_multiplication_factor>1
    number_of_fringes_needed_in_the_frame = fringe_to_fin_size_ratio * number_of_fin_pairs;
    number_of_fringes_in_the_frame = number_of_fringes;
    interpolation_ratio = number_of_fringes_in_the_frame/number_of_fringes_needed_in_the_frame;
    x_current = [-N/2:N/2-1];
    [X_current,Y_current] = meshgrid(x_current);
    sub_window_N = N/2/interpolation_ratio;
     
    %keep a track on total intensity relative to average intensity:
    %(assuming for more than about 8 speckles i'm always at relative intensity=1)
    total_original_energy = sum(sum(abs(speckle_pattern).^2));
    total_sub_pattern_energy = sum(sum(abs(speckle_pattern(N/2+1-floor(sub_window_N/2):N/2+1+floor(sub_window_N/2)+mod(sub_window_N,2),N/2+1-floor(sub_window_N/2):N/2+1+floor(sub_window_N/2)+mod(sub_window_N,2)).^2)));
    relative_energy_of_sub_pattern = total_sub_pattern_energy/total_original_energy;
    relative_expected_energy = sub_window_N/N;
    total_energy_relative_to_expected_mean = relative_energy_of_sub_pattern/relative_expected_energy;    
    
    x_interpolation = linspace(-sub_window_N,sub_window_N-1/interpolation_ratio,N_original);
    [X_interpolation,Y_interpolation] = meshgrid(x_interpolation);
    speckle_pattern = interp2(X_current,Y_current,speckle_pattern,X_interpolation,Y_interpolation,'spline');
else
    total_energy_relative_to_expected_mean = 1; 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%DRAW BEAMS ON SURFACE AND FAR FIELD SPECKLE PATTERN:
if flag_draw_surface_and_far_field_graphs==1
   figure(graph_number)
   imagesc(abs(total_beam));
   title({'beam/beams on the scattering surface',strcat('distance to radius, or speckle to fringe ratio = ',num2str(speckle_to_fin_size_ratio/fringe_to_fin_size_ratio))});
   graph_number=graph_number+1;
   
   figure(graph_number)
   imagesc(abs(speckle_pattern));
   title({'far field speckle pattern',strcat('distance to radius, or speckle to fringe ratio = ',num2str(speckle_to_fin_size_ratio/fringe_to_fin_size_ratio))...
       strcat('speckle to fin size ratio = ',num2str(speckle_to_fin_size_ratio)),strcat('fringe to fin size ratio = ',num2str(fringe_to_fin_size_ratio))});
   graph_number=graph_number+1; 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%DRAW 1D "RAW" MODULATION SIGNAL:
if flag_draw_maskless_modulation_signal==1
   figure(graph_number)
   plot(sum(abs(speckle_pattern).^2,2));
   title({'total energy along the x-axis as a function of y position',...
       strcat('number of fins = ',num2str(number_of_fin_pairs)),...
       strcat('speckle to fin size ratio = ',num2str(speckle_to_fin_size_ratio)),strcat('fringe to fin size ratio = ',num2str(fringe_to_fin_size_ratio))});
   ylabel('intensity[A.U]');
   xlabel('fringe axis[A.U]');
   if flag_draw_maskless_modulation_signal_with_DC==1
      ylim([0,max(sum(abs(speckle_pattern).^2,2))]); 
   end
   graph_number=graph_number+1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%DRAW FFT OF 1D "RAW" MODULATION SIGNAL:
if flag_draw_maskless_modulation_fft==1
   figure(graph_number)
   raw_signal = sum(abs(speckle_pattern).^2,2);
   a=repmat(raw_signal,[20,1]);
   [~,~] = calculate_fft(a,[],0.1,1,length(a),1,1,0,0,graph_number);  
   graph_number=graph_number+2;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MULTIPLY SPECKLE PATTERN BY MASK CREATED TO MAKE "SPECKLE_PATTERN_MODULATED":
if ~isempty(number_of_fin_pairs)
    %ASSIGN AGAIN N=N_ORIGINAL FOR MODULATION SIGNAL CALCULATION!!!!:
    N=N_original;
    [mask] = make_mask_with_integer_number_of_fin_pairs(number_of_fin_pairs,N,0);
    speckle_pattern_modulated = speckle_pattern.*mask;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%DRAW FAR FIELD SPECKLE PATTERN WITH MASK ON IT:
if flag_draw_far_field_with_mask_graph==1
    figure(graph_number)
    imagesc(abs(speckle_pattern_modulated));
    title_str = 'modulator fins on the speckle pattern';
    if flag_speckle_blender==1
       title({title_str,str,strcat('speckle to fin size ratio = ',num2str(speckle_to_fin_size_ratio)),strcat('fringe to fin size ratio = ',num2str(fringe_to_fin_size_ratio))});
    else
       title({title_str,strcat('speckle to fin size ratio = ',num2str(speckle_to_fin_size_ratio)),strcat('fringe to fin size ratio = ',num2str(fringe_to_fin_size_ratio))});
    end
    graph_number=graph_number+1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%BLOCK TO FIND OPTICAL MODULATION OF CHOSEN SPECKLE PATTERN AND MASK:
if flag_draw_modulation_signal==1 || flag_draw_fft_of_signal_modulation==1 || flag_calculate_modulation_signal==1
    if flag_speckle_blender==0
       one_dimensional_speckle_pattern = sum(abs(speckle_pattern).^2,2);
    else
       one_dimensional_speckle_pattern = sum(intensity_speckle_pattern,2);
    end
    raw_modulation = (max(one_dimensional_speckle_pattern)-min(one_dimensional_speckle_pattern))/mean(one_dimensional_speckle_pattern);
    
%     [modulation_signal,modulation] = return_modulation_signal_of_speckle_pattern_and_mask(one_dimensional_speckle_pattern,number_of_fin_pairs,Fs_to_carrier_ratio_for_modulation_calculation,number_of_samples_per_integration,number_of_periods);
    
    Fs_to_carrier_ratio = Fs_to_carrier_ratio_for_modulation_calculation;
    modulation_signal = zeros(Fs_to_carrier_ratio*number_of_periods,1);
    total_energy=0;
    count = 1;
    
    %set new N to input integer number of fins exactly:
    ratio = ceil(N/number_of_fin_pairs);
    N_new = number_of_fin_pairs*ratio;
    
    %IF i choose to move mask then for efficiency i use the 1D speckle pattern:
    if flag_move_speckle_pattern_or_mask==0
        %interpolate mask and 1D speckle pattern so that modulation will be periodic:
        [mask_original] = make_mask_with_integer_number_of_fin_pairs(number_of_fin_pairs,N_new,1);
        mask = mask_original;
        one_dimensional_speckle_pattern = interp1(one_dimensional_speckle_pattern,linspace(1,N,N_new),'spline');

        for tics=1:(Fs_to_carrier_ratio+1)*number_of_periods
            for integration_samples=1:number_of_samples_per_integration_period-1
                current_speckle_pattern = one_dimensional_speckle_pattern'.*mask;
                number_of_pixel_tics = round(count*(N_new)/((number_of_fin_pairs)*Fs_to_carrier_ratio)/number_of_samples_per_integration_period);
                mask = circshift(mask_original,number_of_pixel_tics); 
                if mod(tics,Fs_to_carrier_ratio+1)==0
                   mask=mask_original; 
                   count = 1;
                else
                   count = count+1; 
                end
                total_energy = total_energy + sum(current_speckle_pattern);
            end    
        modulation_signal(tics) = total_energy;
        total_energy = 0;   
        end    
    elseif flag_move_speckle_pattern_or_mask==1
    %IF, on the other hand, i choose to move the speckle pattern to simulate speckle 
    %boiling, i can't use the 1D speckle pattern and i must use the entire 2D pattern.
    %THIS, FOR NOW, IS USABLE ONLY FOR REGULAR TWO BEAMS, NOT FOR SPECKLE BLENDER:
         
        
        
    end
    
    modulation_signal = modulation_signal(1:tics);
    modulation_signal = repmat(modulation_signal,[10,1]);
    modulation = (max(modulation_signal)-min(modulation_signal))/(max(modulation_signal)+min(modulation_signal));
    max_difference = max(modulation_signal)-min(modulation_signal);

    if flag_draw_modulation_signal==1
    figure(graph_number)
    plot(modulation_signal);
    title({strcat('final modulation signal. signal modulation = ',num2str(modulation)),strcat('number of fin pairs = ',num2str(number_of_fin_pairs)),strcat('Fs/Fcarrier =',num2str(Fs_to_carrier_ratio))...
        strcat('speckle to fin size ratio = ',num2str(speckle_to_fin_size_ratio)),strcat('fringe to fin size ratio = ',num2str(fringe_to_fin_size_ratio))});
    graph_number=graph_number+1;
    end
    
    if flag_calculate_fft_of_signal_modulation==1 || flag_draw_fft_of_signal_modulation==1
       modulation_signal = repmat(modulation_signal,[ceil(number_of_samples_for_fft/tics),1]);
       [~,~] = calculate_fft(modulation_signal,[],0.1,1,length(modulation_signal),1,0,0,0,graph_number); 
       graph_number=graph_number+1;
    end
end   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


               
       
%       
% for k=1:10         
%    figure(k)    
%    close gcf;  
% end  
%    
        
    

  

 

