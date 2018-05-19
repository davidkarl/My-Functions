function [flag] = string_does_it_end_with_a_number(original_string)

[~,~,ranges] = string_get_numbers_from_string(original_string);
flag = (ranges(end,end)==length(original_string));

