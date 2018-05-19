function [numbers,indices,ranges] = string_get_numbers_from_string(original_string)

indices = regexp(original_string,'[1-9]+');
numbers_strings = regexp(original_string,'[1-9]+','match');
for k=1:length(numbers_strings)
    ranges = [indices(:), indices(:) + length(numbers_strings{k})];
    numbers = str2num(numbers_strings{k});
end

