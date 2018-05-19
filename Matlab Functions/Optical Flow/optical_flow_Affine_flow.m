function [du,dv] = optical_flow_Affine_flow(mat1,mat2)

%a parameters:
ROI = 1;
region = 1;
roireg = 1;
roitrimmed = 1;
mat1 = 1;
mat2 = 1;
sigmaxy = 1;
mat1_smoothed = 1;
mat2_smoothed = 1;
sampstep = 1;
flow = 1;
resOK = true;




if ~isempty(ROI)
    if isempty(roireg)
        %Trim all arrays to the bounding box to avoid unnecessary work. 
        %Convert to double for regionprops as region may be unconnected
        %bounding_box(1) = upper left corner x
        %bounding_box(2) = upper left corner y
        %bounding_box(3) = width x
        %bounding_box(4) = width y
        bounding_box = regionprops(ROI, 'BoundingBox');
        bounding_box = bounding_box.BoundingBox;
        
        %convert stats to region variable:
        %region(1) = upper left y, region(2) = lower left y
        %region(3) = upper left x, region(4) = upper right x
        region = [ceil(bounding_box(2)), floor(bounding_box(2)+bounding_box(4)), ceil(bounding_box(1)), floor(bounding_box(1)+bounding_box(3))];
        
        %trim array to specified region and get its logical:
        ROI = logical(ROI(region(1):region(2),region(3):region(4)));
         
        %assign variables:
        roireg = region;
        roitrimmed = ROI;
    else
        %assign variables:
        region = roireg;
        ROI = roitrimmed;
    end
elseif isempty(region)
    
    wraps = [false,false];    
    sigmaxy = sigmaxy(:);
    if isscalar(sigmaxy)
        sigmaxy = [sigmaxy sigmaxy];
    elseif isequal(size(sigmaxy), [1 2])
        sigmaxy = sigmaxy([2 1]);
    end
    
    boundopts = {'symmetric' , 'circular'};
    bcons = boundopts(wraps+1);
    regonly = 1;
    imreg = [1,size(mat1,1) 1,size(mat1,2)];
    mrg = ceil(2.6*sigmaxy);
    mrg = [1 -1 1 -1] .* mrg([1 1 2 2]);
    region = imreg + ~wraps([1 1 2 2]) .* mrg;
    
    %compute input region for convolution - expand on all dimensions:
    convreg = region - mrg;    % expand
    if isequal(convreg, imreg)
        convreg = [];   % signal no trimming or padding
    end
    
    mat1 = region;
    region = region + ~wraps([2 2 1 1]) .* [1 -1 1 -1];

end




%Image gradients and x,y coordinate arrays
if isempty(mat1_smoothed)
    mat1_smoothed = gsmooth2(mat1, sigmaxy, region + [-1 1 -1 1], false);
end
if isempty(mat2_smoothed)
    mat2_smoothed = gsmooth2(mat2, sigmaxy, region + [-1 1 -1 1], false);
end



function [mat1, region] = gsmooth2(mat1, sigmaxy, region , wraps)

[sigmaxy, bcons, region, convreg, regonly] = checkinputs(mat1, sigmaxy, region, wraps);
[sigmaxy, bcons, region, convreg, regonly] = checkinputs(mat2, sigmaxy, region, wraps);

if regonly
    mat1 = region;
    mat2 = region;
else
    if ~isempty(convreg)
        mat1 = exindex(mat1, convreg(1):convreg(2), bcons{1},convreg(3):convreg(4), bcons{2});
    end
    if ~isempty(convreg)
        mat2 = exindex(mat2, convreg(1):convreg(2), bcons{1},convreg(3):convreg(4), bcons{2}); 
    end
    mat1 = gsmooth1(mat1, 1, sigmaxy(1));
    mat1 = gsmooth1(mat1, 2, sigmaxy(2));
    mat2 = gsmooth1(mat2, 1, sigmaxy(1));
    mat2 = gsmooth1(mat2, 2, sigmaxy(2));
end

end

% -------------------------------------------------------------------------

function [sigmaxy, bcons, region, convreg, regonly] = checkinputs(mat1, sigmaxy, region, wraps)
% Check arguments and get defaults, plus input/output convolution regions and boundary conditions:

%sigmas argument:
if isscalar(sigmaxy)
    sigmaxy = [sigmaxy sigmaxy];
elseif isequal(size(sigmaxy), [1 2])
    sigmaxy = sigmaxy([2 1]);
end

%wraps argument:
if isscalar(wraps)
    wraps = [wraps wraps];
elseif isequal(size(wraps), [1 2])
    wraps = wraps([2 1]); % xy -> row col for exindex
end
boundopts = {'symmetric' , 'circular'};
bcons = boundopts(wraps+1);

%region argument:
if nargin < 3
    region = [];
end
regonly = strcmp(region, 'region');

imreg = [1, size(mat1,1), 1, size(mat1,2)];   % whole image region
mrg = ceil(2.6*sigmaxy);  % convolution margins
mrg = [1 -1 1 -1] .* mrg([1 1 2 2]);
if isempty(region) || regonly
    % default region - small enough not to need extrapolation - shrink on
    % non-wrapped dimensions
    region = imreg + ~wraps([1 1 2 2]) .* mrg;
elseif strcmp(region, 'same')
    region = imreg;
else
    validateattributes(region, {'double'}, {'real', 'integer', 'size', [1 4]});
end

%compute input region for convolution - expand on all dimensions
convreg = region - mrg;    % expand
if isequal(convreg, imreg)
    convreg = [];   % signal no trimming or padding
end

end

% -------------------------------------------------------------------------

function im = gsmooth1(im, dim, sigma)
% Smooth an image IM along dimension DIM with a 1D Gaussian mask of
% parameter SIGMA

hsize = gausshsize(sigma);  % reasonable truncation

msize = [1 1];
msize(dim) = 2*hsize+1;

if sigma > 0
    mask = fspecial('gauss', msize, sigma);
    im = conv2(im, mask, 'valid');
end

end

% -------------------------------------------------------------------------

function hsize = gausshsize(sigma)
% Default for the limit on a Gaussian mask of parameter sigma.
% Produces a reasonable degree of truncation without too much error.
hsize = ceil(2.6*sigma);
end








[xg, yg, tg] = gradients_xyt(mat1_smoothed, mat2_smoothed, 0, [], false);








%Use origin at centre of image for now - more stable:
[x, y, cx, cy] = centredxy(xg);

%Cut down typing by assembling the arguments:
flowargs = {x, y, xg, yg, tg};

%Resample on bigger grid if required to cut down least-squares solver effort:
if sampstep > 1
    steps = [sampstep,sampstep];
    flowargs = cellfun(@(z) {sample_every_s_step(z, steps)}, flowargs);
end

%Get values in the ROI if there is one, otherwise just convert to column vectors:
if ~isempty(ROI)
    if sampstep > 1
        ROI = sample_every_s_step(ROI, steps);
    end
    flowargs = cellfun(@(z) {z(ROI)}, flowargs);
else
    flowargs = cellfun(@(z) {z(:)}, flowargs);
end

%Assemble and solve the least squares system:
flowl = solvexy(flowargs{:});

% Move origin back to origin of image coordinates
flow = shift(flowl, 1-region(3)-cx, 1-region(1)-cy);
resOK = true;













    function [x, y, cx, cy] = centredxy(im)
        %Makes x and y arrays centred on the centre of the image
        [s1, s2] = size(im);
        cy = (s1-1)/2;
        cx = (s2-1)/2;
        [x, y] = meshgrid(linspace(-cx, cx, s2), linspace(-cy, cy, s1));
    end
    
    
    function [r, w] = logcoords(reg)
        % makes r and w index arrays for the specified region of the
        % log-sampled image
        [r, w] = meshgrid(reg(3)-1:reg(4)-1, reg(1)-1:reg(2)-1);
    end
    
    
    function arr = get_region(arr, reg)
        %trim array to specified region:
        arr = arr(reg(1):reg(2), reg(3):reg(4));
    end
    
    
    function arr = sample_every_s_step(arr, steps)
        %sample every sx'th column and sy'th row, centering the grid:
        sz = size(arr);
        nm1 = floor((sz-1)./steps);
        starts = floor((sz - (steps.*nm1 + 1))/2) + 1;
        arr = arr(starts(2):steps(2):end, starts(1):steps(1):end);
    end


    function affineparams = solvexy(x, y, gx, gy, gt)
        % AFFINE.SOLVEXY first order optic flow parameters -rectilinear
        %   AFFINEPARAMS = AFFINE.SOLVEXY(X, Y, GXS, GYS, GTS) returns
        %   the affine flow parameters vx0, vy0, d, r, s1, s2, as a
        %   struct with those fields. The inputs are the x and y
        %   sample positions and the x, y and t gradients, all as
        %   column vectors.
        
        x_gx = x .* gx;
        y_gx = y .* gx;
        x_gy = x .* gy;
        y_gy = y .* gy;
        A = [gx, gy, x_gx+y_gy, x_gy-y_gx, x_gx-y_gy, y_gx+x_gy];
        
        a = -(A \ gt);
        
        affineparams = struct('vx0', a(1), 'vy0', a(2), ...
            'd', a(3), 'r', a(4), 's1', a(5), 's2', a(6));
    end


    function affineparams = solverw(rmin, nw, r, w, gr, gw, gt)
        % AFFINE.SOLVERW first order optic flow parameters - log-polar
        %   AFFINEPARAMS = AFFINE.SOLVERW(RMIN, NW, R, W, GR, GW, GT)
        %   returns the affine flow parameters vx0, vy0, d, r, s1, s2,
        %   as a struct with those fields. The inputs are the
        %   parameters of the log-polar grid and the gradients measured
        %   on it.
        
        kinv = 2*pi/nw;
        rhoinv = exp(-kinv*r)/rmin;
        theta = w * kinv;
        c = cos(theta);
        s = sin(theta);
        grc = gr .* c;
        grs = gr .* s;
        gwc = gw .* c;
        gws = gw .* s;
        
        gx = (grc - gws) .* rhoinv;
        gy = (grs + gwc) .* rhoinv;
        gs1 = grc.*c - grs.*s - 2*gwc.*s;
        gs2 = 2*grc.*s + gwc.*c - gws.*s;
        
        A = [gx, gy, gr, gw, gs1, gs2];
        
        a = -kinv * (A \ gt);
        
        affineparams = struct('vx0', a(1), 'vy0', a(2), ...
            'd', a(3), 'r', a(4), 's1', a(5), 's2', a(6));
    end



    function m = matrix(f)
        %AFFINE_FLOW.MATRIX converts affine flow structure to matrix
        %   M = AFFINE_FLOW.MATRIX(F) takes an affine flow structure as
        %   returned by AFFINE_FLOW and returns a matrix M such that if
        %   P is a row vector with components [X, Y, 1] representing a
        %   position, P*M is a vector representing the optic flow at
        %   that position.
        m = [f.d+f.s1,  f.s2+f.r;
            f.s2-f.r,  f.d-f.s1;
            f.vx0,     f.vy0];
    end

    function w = warp(f)
        %AFFINE_FLOW.WARP converts a flow to a warp
        %   W = AFFINE_FLOW.WARP(F) takes either a flow structure as
        %   returned by AFFINE_FLOW or a flow matrix as returned by
        %   AFFINE_FLOW2MATRIX and returns a matrix W such that if P is
        %   a row vector with components [X, Y, 1] representing a
        %   position, then P*M is a vector representing the new
        %   position of the vector after one frame of the flow has
        %   occured.
        %
        %   Uses a very simple approximation!
        if isstruct(f)
            f = affine_flow.matrix(f);
        end
        w = f + [eye(2); 0 0];
    end

    function f = shift(f, x0, y0)
        %AFFINE_FLOW.SHIFT shifts the origin of affine flow
        %   F = AFFINE_FLOW.SHIFT(F, X0, Y0) returns affine flow
        %   parameters as returned by AFFINE_FLOW, with the origin
        %   shifted to X0, Y0 relative to the current origin.
        f.vx0 = f.vx0 + x0*(f.d+f.s1) + y0*(f.s2-f.r);
        f.vy0 = f.vy0 + x0*(f.s2+f.r) + y0*(f.d-f.s1);
    end


