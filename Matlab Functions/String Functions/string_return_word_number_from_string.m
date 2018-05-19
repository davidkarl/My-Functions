function [word] = string_return_word_number_from_string(original_string,word_number)

splitted = strsplit(original_string);
word = splitted(word_number);
