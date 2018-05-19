function res = corr2_downsample(mat_in, filter_mat, boundary_conditions_string, downsample_factor_vec, start, stop)
% RES = corrDn(IM, FILT, EDGES, STEP, START, STOP)
%
% Compute correlation of matrices IM with FILT, followed by
% downsampling.  These arguments should be 1D or 2D matrices, and IM
% must be larger (in both dimensions) than FILT.  The origin of filt
% is assumed to be floor(size(filt)/2)+1.
% 
% EDGES is a string determining boundary handling:
%    'circular' - Circular convolution
%    'reflect1' - Reflect about the edge pixels
%    'reflect2' - Reflect, doubling the edge pixels
%    'repeat'   - Repeat the edge pixels
%    'zero'     - Assume values of zero outside image boundary
%    'extend'   - Reflect and invert (continuous values and derivs)
%    'dont-compute' - Zero output when filter overhangs input boundaries
%
% Downsampling factors are determined by STEP (optional, default=[1 1]), 
% which should be a 2-vector [y,x].
% 
% The window over which the convolution occurs is specfied by START 
% (optional, default=[1,1], and STOP (optional, default=size(IM)).
% 
% NOTE: this operation corresponds to multiplication of a signal
% vector by a matrix whose rows contain copies of the FILT shifted by
% multiples of STEP.  See upConv.m for the operation corresponding to
% the transpose of this matrix.


%NOTE: THIS CODE IS NOT ACTUALLY USED! (MEX FILE IS CALLED INSTEAD)

%fprintf(1,'WARNING: You should compile the MEX version of "corrDn.c",\n         found in the MEX subdirectory of matlabPyrTools, and put it in your matlab path.  It is MUCH faster, and provides more boundary-handling options.\n');

%------------------------------------------------------------
% OPTIONAL ARGS:

if exist('boundary_conditions_string','var') 
  if (strcmp(boundary_conditions_string,'reflect1') ~= 1)
    warning('Using REFLECT1 edge-handling (use MEX code for other options).');
  end
end

if ~exist('downsample_factor_vec','var')
	downsample_factor_vec = [1,1];
end	

if ~exist('start','var')
	start = [1,1];
end	

if ~exist('stop','var') 
	stop = size(mat_in);
end	

%------------------------------------------------------------

% Reverse order of taps in filt, to do correlation instead of convolution
filter_mat = filter_mat(size(filter_mat,1):-1:1,size(filter_mat,2):-1:1);

tmp = conv2_reflective_boundary_conditions(mat_in,filter_mat);
res = tmp(start(1):downsample_factor_vec(1):stop(1),start(2):downsample_factor_vec(2):stop(2));

