function [new_string] = string_remove_words_containing_delimiter(original_string,delimiter)

range = string_get_indices_of_words_containing_delimiter(original_string,delimiter);
new_string = string_substitute_range_with_delimiter(original_string,range,'');
