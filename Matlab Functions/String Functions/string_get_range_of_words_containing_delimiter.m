function [relevant_ranges] = string_get_range_of_words_containing_delimiter(original_string,delimiter)

original_words = strsplit(original_string);
[~,relevant_cell_rows] = string_check_if_delimiter_is_in_string_or_string_array(original_words,delimiter);
ranges = string_get_word_ranges_in_string(original_string);
relevant_ranges = ranges(relevant_cell_rows,:);



