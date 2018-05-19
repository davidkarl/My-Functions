function [flag] = string_does_it_end_with_delimiter(original_string,delimiter)

if ~isempty(delimiter)    
    flag = length(original_string) >= length(delimiter) && strcmp(original_string(end-length(delimiter)+1:end), delimiter);    
else
    flag = false;
end

