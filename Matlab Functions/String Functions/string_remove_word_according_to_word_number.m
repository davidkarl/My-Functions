function [new_string] = string_remove_word_according_to_word_number(original_string,word_number)

    [~,~,ranges]=string_get_word_ranges_in_string(original_string);
    new_string = string_substitute_range_with_delimiter(original_string,[ranges(1,1),ranges(1,2)],'');

