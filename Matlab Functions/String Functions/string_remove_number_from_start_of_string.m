function [new_string] = string_remove_number_from_start_of_string(original_string)

flag = string_does_it_start_with_a_number(original_string);
if flag==1
    [~,~,ranges]=string_get_numbers_from_string(original_string);
    new_string = string_substitute_range_with_delimiter(original_string,[ranges(1,1),ranges(1,2)],'');
end
