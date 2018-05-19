function [word] = string_get_word_according_to_word_number(original_string,word_number)

words = strsplit(original_string);
word = words(word_number);
