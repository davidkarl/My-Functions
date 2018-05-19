function [flag] = string_does_it_start_with_a_number(original_string)

[~,~,ranges] = string_get_numbers_from_string(original_string);
flag = (ranges(1,1)==1);
