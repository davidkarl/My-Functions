function [vec_out, mark] = PDTDFB_to_vec(PDTDFB_in)
% PDTDFB2VEC   Convert the output of the PDTDFB into a vector form
%
%       [yind, cfig] = pdtdfb2vec(y)
%
% Input:
%   y:  an output of the PDTDFB
%
% Output:
%   yind :  1-D vector that contains all PDFB coefficients
%   mark :  starting point of each change in band in yind
%
% See also:	PDTDFBDEC, VEC2PDFB, (also PDFB2VEC in Contourlet toolbox)

if iscell(PDTDFB_in{end})
    
    resolutions_vec = 2 : length(PDTDFB_in);
    S = 2*max(size(PDTDFB_in{end}{1}{1}));
    for resolution_counter = 1:(length(PDTDFB_in)-1)
        cfig(resolution_counter) = log2(length(PDTDFB_in{resolution_counter+1}{1}));
    end
    
else
    
    resolutions_vec = 2: length(PDTDFB_in)-1;
    S = size(PDTDFB_in{end}, 1);
    for resolution_counter = 1:(length(PDTDFB_in)-2)
        cfig(resolution_counter) = log2(length(PDTDFB_in{resolution_counter+1}{1}));
    end
    
end
clear in;

% take out the directional subband complex amplitude value
tmp2 = [];
vec_out = [];
% band index
min = 0;
ind = 0;

for resolution_counter = 1:length(resolutions_vec) % for each consider resolution
    
    for direction_counter = 1:length( PDTDFB_in{resolutions_vec(resolution_counter)}{1} )
        min = min+1;
        tmp = PDTDFB_in{resolutions_vec(resolution_counter)}{1}{direction_counter} + ...
                                       1j*PDTDFB_in{resolutions_vec(resolution_counter)}{2}{direction_counter};
        
        %first column is the starting point of the subband
        mark(min,1) = size(vec_out,1);
        %second column is the row size of the subband
        mark(min,2) = size(tmp,1);
        %third column is the column size of the subband
        mark(min,3) = size(tmp,2);
        %fourth column resolution the subband
        mark(min,4) = resolutions_vec(resolution_counter);
        %fifth column direction the subband
        mark(min,5) = direction_counter;
        
        % [inc, inr] = meshgrid(1:Stmp(2), 1:Stmp(1));
        
        %
        % tmp3 = [(tmp(:));
        
        vec_out = [vec_out; tmp(:)];
    end
end
