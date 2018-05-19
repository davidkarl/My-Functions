function [new_string] = string_remove_delimiter_from_string(original_string,delimiter)

% cell_array_without_delimiter = strsplit(original_string,delimiter); 
% new_string = strjoin(cell_array_without_delimiter);

new_string = strrep(original_string,delimiter,'');

