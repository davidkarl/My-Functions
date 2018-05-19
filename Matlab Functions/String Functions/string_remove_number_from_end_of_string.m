function [new_string] = string_remove_number_from_end_of_string(original_string)

flag = string_does_it_end_with_a_number(original_string);
if flag==1
    [~,~,ranges]=string_get_numbers_from_string(original_string);
    new_string = string_substitute_range_with_delimiter(original_string,[ranges(end,1),ranges(end,2)],'');
end
