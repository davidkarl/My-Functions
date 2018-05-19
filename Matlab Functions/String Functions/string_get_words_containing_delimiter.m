function [words] = string_get_words_containing_delimiter(original_string,delimiter)

original_words = strsplit(original_string);
flags = string_check_if_delimiter_is_in_string(original_words,delimiter);
words = original_words(flags);
