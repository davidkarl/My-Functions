function [res] = make_2D_zone_plate(mat_size, amplitude, phase_offset)
% IM = mkZonePlate(SIZE, AMPL, PHASE)
%
% Make a "zone plate" image:
%     AMPL * cos( r^2 + PHASE)
% SIZE specifies the matrix size, as for zeros().
% AMPL (default = 1) and PHASE (default = 0) are optional.

mat_size = mat_size(:);
if (size(mat_size,1) == 1)
    mat_size = [mat_size,mat_size];
end

max_mat_size = max(mat_size(1),mat_size(2));

%------------------------------------------------------------
% OPTIONAL ARGS:

if ~exist('amplitude','var')
    amplitude = 1;
end

if ~exist('phase_offset','var')
    phase_offset = 0;
end

%------------------------------------------------------------

res = amplitude * cos( (pi/max_mat_size) * make_2D_ramp(mat_size,2) + phase_offset );

