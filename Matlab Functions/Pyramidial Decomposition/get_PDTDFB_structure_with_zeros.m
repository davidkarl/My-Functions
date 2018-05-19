function y = get_PDTDFB_structure_with_zeros(mat_size, level_of_decomposition, flag_residual)
% MKZERO_PDTDFB   Make a structure of pdtdfb with all zeros
%
%   y = mkZero_pdtdfb(S, lev)
%
% Input:
%   y:	    Size of the image that structure y represent
%   lev:    Level of decomposition, see PDTDFBDEC
%   res:    Optinal, Residual band exist or not
% Output:
%   y:	    zero data structure
%
% See also: PDTDFBDEC, PDTDFBREC

N = length(level_of_decomposition);

%lowest resolution level:
if max(size(mat_size)) == 1
    mat_size = [mat_size , mat_size];
end

y{1} = zeros(mat_size(1)/2^N , mat_size(2)/2^N);


for level_counter = 2 : N+1
    current_level_size = mat_size./2^(N+1-level_counter);
    
    for in = 1 : 2^(level_of_decomposition(level_counter-1)-1);
        s1 = current_level_size(1)/2^(level_of_decomposition(level_counter-1)-1);
        s2 = current_level_size(2)/2;
        y{level_counter}{1}{in} = zeros(s1, s2);
        y{level_counter}{2}{in} = zeros(s1, s2);
    end
    
    for in = 2^(level_of_decomposition(level_counter-1)-1) + 1 : 2^(level_of_decomposition(level_counter-1));
        s1 = current_level_size(1)/2;
        s2 = current_level_size(2)/2^(level_of_decomposition(level_counter-1)-1);
        y{level_counter}{1}{in} = zeros(s1, s2);
        y{level_counter}{2}{in} = zeros(s1, s2);
    end
    
end

if ~exist('res','var')
    flag_residual = 0;
end

if flag_residual
    % residual level
    y{N + 2} = zeros(mat_size(1), mat_size(2));
end
