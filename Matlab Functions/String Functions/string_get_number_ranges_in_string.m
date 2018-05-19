function [ranges] = string_get_number_ranges_in_string(original_string)

indices = regexp(original_string,'[1-9]+');
numbers_strings = regexp(original_string,'[1-9]+','match');
for k=1:length(numbers_strings)
    ranges = [indices(:), indices(:) + length(numbers_strings{k})];
end
