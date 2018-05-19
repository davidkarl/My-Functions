function [c, s] = PDFB_to_vec(PDFB_in)
% PDFB2VEC   Convert the output of the PDFB into a vector form
%
%       [c, s] = pdfb2vec(y)
%
% Input:
%   y:  an output of the PDFB
%
% Output:
%   c:  1-D vector that contains all PDFB coefficients
%   s:  structure of PDFB output, which is a four-column matrix.  Each row
%       of s corresponds to one subband y{l}{d} from y, in which the first two
%       entries are layer index l and direction index d and the last two
%       entries record the size of y{l}{d}.
%
% See also:	PDFBDEC, VEC2PDFB

number_of_levels = length(PDFB_in);

%Save the structure of y into s:
s(1, :) = [1, 1, size(PDFB_in{1})];

%Used for row index of s:
ind = 1;

%Build modified extended index matrix:
for level_counter = 2:number_of_levels
    number_of_directions = length(PDFB_in{level_counter});
    
    for direction_counter = 1:number_of_directions
        s(ind + direction_counter, :) = ...
                                [level_counter, direction_counter, size(PDFB_in{level_counter}{direction_counter})];
    end
    
    ind = ind + number_of_directions;
end

%The total number of PDFB coefficients:
total_number_of_PDFB_coefficients = sum(prod(s(:, 3:4), 2));

%Assign the coefficients to the vector c:
c = zeros(1, total_number_of_PDFB_coefficients);

%Variable that keep the current position:
pos = prod(size(PDFB_in{1}));

%Lowpass subband:
c(1:pos) = PDFB_in{1}(:);

%Bandpass subbands:
for level_counter = 2:number_of_levels    
    for direction_counter = 1:length(PDFB_in{level_counter})
        ss = prod(size(PDFB_in{level_counter}{direction_counter}));
        c(pos+[1:ss]) = PDFB_in{level_counter}{direction_counter}(:);
        pos = pos + ss;
    end
end



