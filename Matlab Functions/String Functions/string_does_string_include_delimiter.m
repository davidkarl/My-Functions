function [flag] = string_does_string_include_delimiter(original_string,delimiter)

flag=~(isempty(strfind(original_string,delimiter)));
