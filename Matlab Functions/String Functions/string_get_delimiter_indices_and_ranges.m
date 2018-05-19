function [indices,ranges] = string_get_delimiter_indices_and_ranges(original_string,delimiter)

indices = strfind(original_string,delimiter);
ranges = [indices(:),indices(:)+length(delimiter)];
