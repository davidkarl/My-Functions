function [res] = make_2D_sine(mat_size, period_in_pixels, direction_or_amplitude, amplitude_or_phase, phase_origin, origin)
% IM = mkSine(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN)
%      or
% IM = mkSine(SIZE, FREQ, AMPLITUDE, PHASE, ORIGIN)
%
% Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
% containing samples of a 2D sinusoid, with given PERIOD (in pixels),
% DIRECTION (radians, CW from X-axis, default = 0), AMPLITUDE (default
% = 1), and PHASE (radians, relative to ORIGIN, default = 0).  ORIGIN
% defaults to the center of the image.
%
% In the second form, FREQ is a 2-vector of frequencies (radians/pixel).


%-----------------------------------------------------------
% OPTIONAL ARGS:

if (prod(size(period_in_pixels)) == 2)
    %if i got 1 frequency for each axis then direction is already decided:
    sine_frequency = norm(period_in_pixels);
    sine_direction = atan2(period_in_pixels(1),period_in_pixels(2));
    if (exist('direction_or_amplitude','var') == 1)
        sine_amplitude = direction_or_amplitude;
    else
        sine_amplitude = 1;
    end
    if (exist('amplitude_or_phase','var') == 1)
        sine_phase = amplitude_or_phase;
    else
        sine_phase = 0;
    end
    if (exist('phase_origin','var') == 1)
        origin = phase_origin;
    end
    if (exist('origin','var') == 1)
        error('Too many arguments for (second form) of mkSine');
    end
else
    sine_frequency = 2*pi/period_in_pixels;
    if (exist('direction_or_amplitude','var') == 1)
        sine_direction = direction_or_amplitude;
    else
        sine_direction = 0;
    end
    if (exist('amplitude_or_phase','var') == 1)
        sine_amplitude = amplitude_or_phase;
    else
        sine_amplitude = 1;
    end
    if (exist('phase_origin','var') == 1)
        sine_phase = phase_origin;
    else
        sine_phase = 0;
    end
    if (exist('origin','var') == 1)
        origin = origin;
    end
end

%------------------------------------------------------------

if (exist('origin') == 1)
    res = sine_amplitude*sin(make_2D_ramp(mat_size, sine_direction, sine_frequency, sine_phase, origin));
else
    res = sine_amplitude*sin(make_2D_ramp(mat_size, sine_direction, sine_frequency, sine_phase));
end
