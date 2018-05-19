function [bla] = Riesz_Transform_Phase_magnification_PSEUDOCODE(amplification_factor,low_cutoff,high_cutoff,Fs)
%Riesz Pyramid for fast phase based video magnification PSEUDOCODE

%Initializes spatial smoothing kernel and temporal filtering coefficients.

%compute an IIR temporal filter coefficients:
nyquist_frequency = Fs/2;
temporal_filter_order = 1;
[B,A] = get_butterworth_filter_coefficients(temporal_filter_order, low_cutoff/nyquist_frequency,high_cutoff/nyquist_frequency);

%compute convolution kernel for spatial blurring kernel used during
%queternionic phase denoising step:
gaussian_kernel_sigma = 2; %[pixels]
gaussian_kernel = get_gaussian_kernel(gaussian_kernel_sigma);

%Initialization of variables before main loop (התחלנו!) 
%this initialization is equivalent to assuming the motions are zero before the video stats:
previous_frame = get_first_frame_from_video();
[previous_laplacian_pyramid, previous_riesz_x, previous_riesz_y] = compute_riesz_pyramid(previous_frame);
number_of_levels = numel(previous_laplacian_pyramid) - 1; %don't include lowpass residual

for k = 1:number_of_levels
   %Initializate current value of quaternionic phase. 
   %each coefficients has a two element quaternionic phase that is 
   %defined as phase time*(cos(orientation,sin(orientation)). 
   phase_cos{k} = zeros(size(previous_laplacian_pyramid{k}));
   phase_sin{k} = zeros(size(previous_laplacian_pyramid{k}));
   
   %Initializes IIR temporal filter values. these values are used during temporal filtering.
   %See the function IIRTemporalFilter for more details.
   %The initialization is a zero motion boundary condition at the beginning of the video.
   register0_cos{k} = zeros(size(previous_laplacian_pyramid{k}));
   register1_cos{k} = zeros(size(previous_laplacian_pyramid{k}));
   register0_sin{k} = zeros(size(previous_laplacian_pyramid{k}));
   register1_sin{k} = zeros(size(previous_laplacian_pyramid{k}));
end


%MAIN LOOP.
%it is executed on new frames from the video and runs untill stopped.
while running
    current_frame = get_next_frame_from_video();
    [current_laplacian_pyramid,current_riesz_x,current_riesz_y] = compute_riesz_pyramid(current_frame);
    
    %we compute a laplacian pyramid of the motion magnified frame first and then collapse it at the end.
    %the processing in the following loop is processed on each level of the riesz pyramid independently.
    for k = 1:number_of_levels
        
        %compute quaternionic phase difference between current riesz
        %pyramid coefficients and previous riesz pyramid coefficients:
        [phase_difference_cos, phase_difference_sin, amplitude] = ...
                                    compute_phase_difference_and_amplitude(current_laplacian_pyramid{k},...
                                                                           current_riesz_x{k},...
                                                                           current_riesz_y{k},...
                                                                           previous_laplacian_pyramid{k},...
                                                                           previous_riesz_x{k},...
                                                                           previous_riesz_y{k});
                                                                       
                                                     
        %adds the quaternionic phase difference to the current value of the quaternioni phase.
        %computing the current value of the phase in this way is equivalent to phase unwrapping:
        phase_cos{k} = phase_cos{k} + phase_difference_cos;
        phase_sin{k} = phase_sin{k} + phase_difference_sin;
        
        %temporally filter the quaternionic phase using current value and stored information:
        [phase_filtered_cos, register0_cos{k}, register1_cos{k}] = ...
            iir_temporal_filter(B,A, phase_cos{k}, register0_cos{k}, register1_cos{k});
        [phase_filtered_sin, register0_sin{k}, register1_sin{k}] = ...
            iir_temporal_filter(B,A, phase_sin{k}, register0_sin{k}, register1_sin{k});
        
         
        %spatially blur the temporally filtered quaternionic phase signals.
        %this is not an optional step. in addition to denoising, it smooths
        %out errors made during the various approximations:
        phase_filtered_cos = amplitude_weighted_blur(phase_filtered_cos, amplitude, gaussian_kernel);
        phase_filtered_sin = amplitude_weighted_blur(phase_filtered_sin, amplitude, gaussian_kernel);
        
        %The motion magnified pyramid is computed by phase shifting the
        %input pyramid by the spatio-temporally filtered quaternonic phase
        %and TAKING THE REAL PART:
        phase_magnified_filtered_cos = amplification_factor * phase_filtered_cos;
        phase_magnified_filtered_sin = amplification_factor * phase_filtered_sin;
        motion_magnified_laplacian_pyramid{k} = ...
            phase_shift_coefficient_real_part(current_laplacian_pyramid{k},...
                                              current_riesz_x{k},...
                                              current_riesz_y{k},...
                                              phase_magnified_filtered_cos,...
                                              phase_magnified_filtered_sin); 
        
    end %number_of_levels loop
    
    %take lowpass residual from curent frame's lowpass residual and collapse pyramid:
    motion_magnified_laplacian_pyramid{number_of_levels+1} = ...
        current_laplacian_pyramid{number_of_levels+1};
    motion_magnified_frame = collapse_laplacian_pyramid{motion_magnified_laplacian_pyramid);
    
    %write or display the motion magnified frame:
    write_magnified_frame(motion_magnified_rame);
    
    %prepare for next iteration of loop:
    previous_laplacian_pyramid = current_laplacian_pyramid;
    previous_riesz_x = current_riesz_x;
    previous_riesz_y = current_riesz_y;
    
    
end %while(running) loop

end


function [laplacian_pyramid,riesz_x,riesz_y] = compute_riesz_pyramid(current_frame)
    %compute the riesz pyramid of a 2D frame. this is done by first
    %computing the laplcian pyramid of the frame and then computing the
    %approxiate riesz transform of each level that is not the lowpass
    %residual. the result is stored as an array of grayscale frames.
    %corresponding locations in the result correspond to the real, i&j
    %components of riesz pyramid coefficients.
    
    laplacian_pyramid = compute_laplacian_pyramid(grayscale_frame);
    number_of_levels = numel(laplacian_pyramid)-1;
    
    %the approximate riesz transform of each level that is not the lowpass
    %residual is computed:
    kernel_x = [0,0,0;...
               0.5,0,-0.5;...
               0,0,0];
    kernel_y = [0,0.5,0;...
               0,0,0;...
               0,-0.5,0];
    
    for k = 1:number_of_levels
       riesz_x{k} = conv2(laplacian_pyramid{k},kernel_x);
       riesz_y{k} = conv2(laplacian_pyramid{k},kernel_y);
    end
    
end


function [phase_difference_cos, phase_difference_sin, amplitude] = ...
                                compute_phase_difference_and_amplitude(current_real_pyramid_band,...
                                                                       current_riesz_x,...
                                                                       current_riesz_y,...
                                                                       previous_real_pyramid_band,...
                                                                       previous_riesz_x,...
                                                                       previous_riesz_y)
    %compute quaternionic phase difference between current frame and
    %previous frame. this is done by dividing the coefficients of the
    %current frame and the previous frame and then taking imaginary part of
    %the quaternionic logarithm. we asume the ORIENTATION at a point is
    %roughly constant to simplify the calculation.
    
    %q_current = current_real + i*current_x + j*current_y
    %q_previous = previous_Real + i*previous_x +j*previous_y
    %we want to compute the phase difference, which is the phase of q_current/q_previous
    %--> this is equal to: q_current*conjugate(q_previous)/||q_previous||^2
    %phase is invariant to scalar multiples, so we want the phase of
    %q_current*conjugate(previous).
    %UNDER THE CONSTANT ORIENTATION ASSUMPTION, we can assume the fourth
    %component of the prduct is zero.
    
    %CHECK IF THIS CAN BE DONE MORE EFFICIENTLY:
    q_conj_prod_real = current_real_pyramid_band.*previous_real_pyramid_band + ...
                       current_riesz_x.*previous_riesz_x + ...
                       current_riesz_y.*previous_riesz_y;
    q_conj_prod_x = -current_real_pyramid_band.*previous_riesz_x + previous_real_pyramid_band.*current_riesz_x;
    q_conj_prod_y = -current_real_pyramid_band.*previous_riesz_y + previous_real_pyramid_band.*current_riesz_y;
    
    %now we take the quaternioni logarithm of this.
    %only the imaginary part corresponds to quaternionic phase.
    q_conj_prod_amplitude = sqrt(q_conj_prod_real.^2 + q_conj_prod_x.^2 + q_conj_prod_y.^2);
    phase_difference = acos(q_conj_prod_real./q_conj_prod_amplitude);
    cos_orientation = q_conj_prod_x ./ sqrt(q_conj_prod_x.^2+q_conj_prod_y.^2);
    sin_orientation = q_conj_prod_y ./ sqrt(q_conj_prod_x.^2+q_conj_prod_y.^2);
    
    %this is the quaternionic phase:
    phase_difference_cos = phase_difference .* cos_orientation;
    phase_difference_sin = phase_difference .* sin_orientation;
    
    %under the assumption that changes are small between frames, we can
    %assume that the amplitude of both coefficients is the same. so, to
    %compute the amplitude of one coefficient, we just take the square root
    %of their conjugate product:
    amplitude = sqrt(q_conj_prod_amplitude);
    
end




function [temporally_filtered_phase,register0,register1] = iir_temporal_filter(B,A,phase,register0,register1)
    %temporaly filters phase with IIR filter with coefficients B,A.
    %given current phase value and value of previously computed registers,
    %computes current temporally filtered phase value and updates
    %registers. assumes filter given by B,A is a first order IIR filter, so
    %that B and A have 3 coefficients each. also, assumes A(1)=1.
    %computation is direct form type 2 filter.
    temporally_filtered_phase = B(1)*phase + register0;
    register0 = B(2)*phase + register1 - A(2)*temporally_filtered_phase;
    register1 = B(3)*phase             - A(3)*temporally_filtered_phase;
end


function [spatially_smooth_temporally_filtered_phase] = ...
                        amplitude_weighted_blur(temporally_filtered_phase,amplitude,blur_kernel)
    %spatially blurs phase, weighted by amplitude:
    
    %MAYBE I CAN GET AMPLITUDE AND THEREFORE NOT CALCULATE AMPLITUDE
    %CONVOLUTION IN DENOMINATOR IN EVERY FRAME???
    
    %MAYBE USE IRREGAULR SAMPLING LIFTING FILTERING STRCUTURE AND
    %INTERPOLATE LOW AMPLITUDE PARTS OF THE IMAGE
    denominator = conv2(amplitude, blur_kernel);
    numerator = conv2(temporally_filtered_phase.*amplitude, blur_kernel);
    spatiialy_smooth_temporally_filtered_phase = numerator./denominator;
end

function [result] = phase_shift_coefficient_real_part(riesz_real,riesz_x,riesz_y,phase_cos,phase_sin)
    %phase shifts a riesz pyramid coefficient and returns the real part of
    %the resulting coefficient. the input coefficient is a three element
    %quaternioin. the phase is a two element imaginary quaternion.
    %the phase is exponentiated and then the result is multiplied by the
    %first coefficient.
    
    %quaternion exponentiation:
    phase_magnitude = sqrt(phase_cos.^2 + phase_sin.^2); 
    exp_phase_real = cos(phase_magnitude);
    exp_phase_x = phase_cos./phase_magnitude.*sin(phase_magnitude);
    exp_phase_y = phase_sin./phase_magnitude.*sin(phase_magnitude);
    
    %quaternion multiplication (just real part):
    result = exp_phase_real.*riesz_real - exp_phase_x.*riesz_x - exp_phase_y.*riesz_y;
    
end
















