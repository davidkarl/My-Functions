function [flag] = string_check_if_delimiter_is_in_range(original_string,range,delimiter)

flag = ~(isempty(find(original_string(range(1):range(2)),delimiter)));

