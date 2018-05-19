function result = upsample_inserting_zeros_convolve(mat_in,filter_mat,boundary_conditions_string,upsampling_factor_vec,start_indices,stop_indices,res)

% THIS CODE IS NOT ACTUALLY USED! (MEX FILE IS CALLED INSTEAD)

%fprintf(1,'WARNING: You should compile the MEX version of "upConv.c",\n         
%found in the MEX subdirectory of matlabPyrTools, and put it in your matlab path.  
%It is MUCH faster, and provides more boundary-handling options.\n');


% RES = upConv(IM, FILT, EDGES, STEP, START, STOP, RES)
%
% Upsample matrix IM, followed by convolution with matrix FILT.  These
% arguments should be 1D or 2D matrices, and IM must be larger (in
% both dimensions) than FILT.  The origin of filt
% is assumed to be floor(size(filt)/2)+1.
%
% EDGES is a string determining boundary handling:
%    'circular' - Circular convolution
%    'reflect1' - Reflect about the edge pixels
%    'reflect2' - Reflect, doubling the edge pixels
%    'repeat'   - Repeat the edge pixels
%    'zero'     - Assume values of zero outside image boundary
%    'extend'   - Reflect and invert
%    'dont-compute' - Zero output when filter overhangs OUTPUT boundaries
%
% Upsampling factors are determined by STEP (optional, default=[1 1]),
% a 2-vector [y,x].
% 
% The window over which the convolution occurs is specfied by START 
% (optional, default=[1,1], and STOP (optional, default = 
% step .* (size(IM) + floor((start-1)./step))).
%
% RES is an optional result matrix.  The convolution result will be 
% destructively added into this matrix.  If this argument is passed, the 
% result matrix will not be returned. DO NOT USE THIS ARGUMENT IF 
% YOU DO NOT UNDERSTAND WHAT THIS MEANS!!
% 
% NOTE: this operation corresponds to multiplication of a signal
% vector by a matrix whose columns contain copies of the time-reversed
% (or space-reversed) FILT shifted by multiples of STEP.  See corrDn.m
% for the operation corresponding to the transpose of this matrix.


%------------------------------------------------------------
% OPTIONAL ARGS:
if exist('boundary_conditions_string','var') 
  if (strcmp(boundary_conditions_string,'reflect1') ~= 1)
    warning('Using REFLECT1 edge-handling (use MEX code for other options).');
  end
end

if ~exist('upsampling_factor_vec','var')
  upsampling_factor_vec = [1,1];
end	

if ~exist('start_indices','var')
  start_indices = [1,1];
end	

%A multiple of step:
if ~exist('stop_indices','var')
   upsampling_factor_vec .* (floor((start_indices-ones(size(start_indices)))./upsampling_factor_vec) + size(mat_in));
end	

%Make sure the convolution indices cover the original matrix:
if ( ceil((stop_indices(1)-start_indices(1)+1) / upsampling_factor_vec(1)) ~= size(mat_in,1) )
  error('Bad Y result dimension');
end
if ( ceil((stop_indices(2)-start_indices(2)+1) / upsampling_factor_vec(2)) ~= size(mat_in,2) )
  error('Bad X result dimension');
end

if ~exist('res','var')
  res = zeros(stop_indices-start_indices+1);
end	

%------------------------------------------------------------

%Insert Zeros before convolution
tmp = zeros(size(res));
tmp(start_indices(1):upsampling_factor_vec(1):stop_indices(1),...
    start_indices(2):upsampling_factor_vec(2):stop_indices(2)) = mat_in;

%Convolve
result = conv2_reflective_boundary_conditions(tmp,filter_mat) + res;




