function [res] = make_2D_square_wave(mat_size, period_in_pixels, direction_or_amplitude,...
    amplitude_or_phase, phase_offset, origin, transition_width)
% IM = mkSquare(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN, TWIDTH)
%      or
% IM = mkSine(SIZE, FREQ, AMPLITUDE, PHASE, ORIGIN, TWIDTH)
%
% Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
% containing samples of a 2D square wave, with given PERIOD (in
% pixels), DIRECTION (radians, CW from X-axis, default = 0), AMPLITUDE
% (default = 1), and PHASE (radians, relative to ORIGIN, default = 0).
% ORIGIN defaults to the center of the image.  TWIDTH specifies width
% of raised-cosine edges on the bars of the grating (default =
% min(2,period/3)).
%
% In the second form, FREQ is a 2-vector of frequencies (radians/pixel).

% TODO: Add duty cycle.


%------------------------------------------------------------
% OPTIONAL ARGS:

if numel(period_in_pixels) == 2
    frequency = norm(period_in_pixels);
    direction = atan2(period_in_pixels(1),period_in_pixels(2));
    if exist('direction_or_amplitude','var')
        amplitude = direction_or_amplitude;
    else
        amplitude = 1;
    end
    if exist('amplitude_or_phase','var')
        phase = amplitude_or_phase;
    else
        phase = 0;
    end
    if exist('phase_offset','var')
        origin = phase_offset;
    end
    if exist('origin','var')
        transition = origin;
    else
        transition = min(2,2*pi/(3*frequency));
    end
    if exist('transition_width','var')
        error('Too many arguments for (second form) of mkSine');
    end
else
    frequency = 2*pi/period_in_pixels;
    if exist('direction_or_amplitude','var')
        direction = direction_or_amplitude;
    else
        direction = 0;
    end
    if exist('amplitude_or_phase','var')
        amplitude = amplitude_or_phase;
    else
        amplitude = 1;
    end
    if exist('phase_offset','var')
        phase = phase_offset;
    else
        phase = 0;
    end
    if exist('origin','var')
        origin = origin;
    end
    if exist('transition_width','var')
        transition = transition_width;
    else
        transition = min(2,2*pi/(3*frequency));
    end
    
end

%------------------------------------------------------------

if exist('origin')
    res = make_2D_ramp(mat_size, direction, frequency, phase, origin) - pi/2;
else
    res = make_2D_ramp(mat_size, direction, frequency, phase) - pi/2;
end

[Xtbl,Ytbl] = make_raised_cosine(transition*frequency,pi/2,[-amplitude amplitude]);

res = apply_point_operation_to_image(abs(mod(res+pi, 2*pi)-pi),Ytbl,Xtbl(1),Xtbl(2)-Xtbl(1),0);

% OLD threshold version:
%res = amplitude * (mod(res,2*pi) < pi);
