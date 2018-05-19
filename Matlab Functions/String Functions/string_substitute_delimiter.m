function [new_string] = string_substitute_delimiter_with_another_string(original_string,delimiter,delimiter_substitution)

new_string = strrep(original_string,delimiter,delimiter_substitution);


