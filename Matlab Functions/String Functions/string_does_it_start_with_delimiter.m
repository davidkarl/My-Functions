function [flag] = string_does_it_start_with_delimiter(original_string,delimiter)

if ~isempty(delimiter)    
    flag = length(original_string) >= length(delimiter) && strcmp(original_string(1:length(delimiter)), delimiter);    
else
    flag = false;
end
