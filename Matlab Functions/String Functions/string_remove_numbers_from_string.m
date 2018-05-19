function [new_string] = string_remove_numbers_from_string(original_string)

ranges = string_get_number_ranges_in_string(original_string);
new_string = string_remove_range_from_string(original_string,ranges);

