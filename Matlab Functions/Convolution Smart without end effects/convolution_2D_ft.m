function [convolved_mat] = convolution_2D_ft(mat1,mat2,flag_use_padding,varargin)
%assuming both matrices are rectangular AND with even number of parts,
%i only return a mat of size equal to mat1,

if nargin==2
    flag_use_padding=1;
end

if flag_use_padding==1
    [mat1,mat2,indices] = pad_arrays_for_convolution(mat1,mat2,1);
end

delta=1;
U1 = ft2(mat1, delta); % DFTs of signals
U2 = ft2(mat2, delta);
delta_f = 1/(size(mat1,1)*delta); % frequency grid spacing [m]
convolved_mat = real(ift2(U1 .* U2, delta_f));

if flag_use_padding==1
    % trim to correct output size
    convolved_mat = convolved_mat(indices(1):indices(2),indices(3):indices(4)); 
end
         
