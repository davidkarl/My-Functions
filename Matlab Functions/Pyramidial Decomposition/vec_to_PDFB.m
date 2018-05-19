function y = vec_to_PDFB(c, s)
% VEC2PDFB   Convert the vector form to the output structure of the PDFB
%
%       y = vec2pdfb(c, s)
%
% Input:
%   c:  1-D vector that contains all PDFB coefficients
%   s:  structure of PDFB output
%
% Output:
%   y:  PDFB coefficients in cell vector format that can be used in pdfbrec
%
% See also:	PDFB2VEC, PDFBREC

%Copy the coefficients from c to y according to the structure s:
number_of_levels = s(end, 1);      % number of pyramidal layers
y = cell(1, number_of_levels);

%Variable that keep the current position:
pos = prod(s(1, 3:4));

%Insert first subband:
y{1} = reshape(c(1:pos), s(1, 3:4));

%Used for row index of s:
ind = 1;

for level_counter = 2:number_of_levels
    %Number of directional subbands in this layer:
    number_of_directions = length(find(s(:, 1) == level_counter));

    y{level_counter} = cell(1, number_of_directions);
    
    for direction_counter = 1:number_of_directions
        %Size of this subband:
        p = s(ind + direction_counter, 3);
        q = s(ind + direction_counter, 4);
        total_subband_size = p * q;
        
        %Assign proper coefficients to PDFB structure:
        y{level_counter}{direction_counter} = reshape(c(pos+[1:total_subband_size]), [p, q]);
        pos = pos + total_subband_size;
    end
    
    ind = ind + number_of_directions;
end




