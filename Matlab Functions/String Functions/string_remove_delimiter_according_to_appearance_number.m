function [new_string] = string_remove_delimiter_according_to_appearance_number(original_string,delimiter,appearance_number)

[indices,ranges] = string_get_delimiter_indices_and_ranges(original_string,delimiter);
new_string = original_string;
new_string(ranges(appearance_number,1):ranges(appearance_number,2))='';

