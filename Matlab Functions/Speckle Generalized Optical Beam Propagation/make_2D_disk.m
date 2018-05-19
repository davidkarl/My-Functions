function [res] = make_2D_disk(mat_size, disk_radius, origin, soft_threshold_transition_width, inside_and_outside_disk_values)
% IM = mkDisc(SIZE, RADIUS, ORIGIN, TWIDTH, VALS)
%
% Make a "disk" image.  SIZE specifies the matrix size, as for
% zeros().  RADIUS (default = min(size)/4) specifies the radius of
% the disk.  ORIGIN (default = (size+1)/2) specifies the
% location of the disk center.  TWIDTH (in pixels, default = 2)
% specifies the width over which a soft threshold transition is made.
% VALS (default = [0,1]) should be a 2-vector containing the
% intensity value inside and outside the disk.


if (nargin < 1)
    error('Must pass at least a size argument');
end

mat_size = mat_size(:);
if (size(mat_size,1) == 1)
    mat_size = [mat_size mat_size];
end

%------------------------------------------------------------
% OPTIONAL ARGS:

if ~exist('disk_radius','var')
    disk_radius = min(mat_size(1),mat_size(2))/4;
end

if ~exist('origin','var')
    origin = (mat_size+1)./2;
end

if ~exist('soft_threshold_transition_width','var')
    soft_threshold_transition_width = 2;
end

if ~exist('inside_and_outside_disk_values','var')
    inside_and_outside_disk_values = [1,0];
end

%------------------------------------------------------------

res = make_2D_ramp(mat_size,1,origin);

if (abs(soft_threshold_transition_width) < realmin)
    res = inside_and_outside_disk_values(2) + ...
        (inside_and_outside_disk_values(1) - inside_and_outside_disk_values(2)) * (res <= disk_radius);
else
    [Xtbl,Ytbl] = make_raised_cosine(soft_threshold_transition_width, disk_radius, [inside_and_outside_disk_values(1), inside_and_outside_disk_values(2)]);
    res = pointOp(res, Ytbl, Xtbl(1), Xtbl(2)-Xtbl(1), 0);
    %
    % OLD interp1 VERSION:
    %  res = res(:);
    %  Xtbl(1) = min(res);
    %  Xtbl(size(Xtbl,2)) = max(res);
    %  res = reshape(interp1(Xtbl,Ytbl,res), sz(1), sz(2));
    %
end


