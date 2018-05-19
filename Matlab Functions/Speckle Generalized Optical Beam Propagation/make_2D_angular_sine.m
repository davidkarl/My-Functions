function [res] = make_2D_angular_sine(mat_size, angular_frequency, amplitude, phase_offset, origin)
% IM = mkAngularSine(SIZE, HARMONIC, AMPL, PHASE, ORIGIN)
%
% Make an angular sinusoidal image:
%     AMPL * sin( HARMONIC*theta + PHASE),
% where theta is the angle about the origin.
% SIZE specifies the matrix size, as for zeros().
% AMPL (default = 1) and PHASE (default = 0) are optional.

mat_size = mat_size(:);
if (size(mat_size,1) == 1)
    mat_size = [mat_size,mat_size];
end

mxsz = max(mat_size(1),mat_size(2));

%------------------------------------------------------------
% OPTIONAL ARGS:

if ~exist('angular_frequency','var')
    angular_frequency = 1;
end

if ~exist('amplitude','var')
    amplitude = 1;
end

if ~exist('phase_offset','var')
    phase_offset = 0;
end

if ~exist('origin','var')
    origin = (mat_size+1)/2;
end

%------------------------------------------------------------

res = amplitude * sin(angular_frequency*make_2D_angles_meshgrid(mat_size,phase_offset,origin) + phase_offset);

