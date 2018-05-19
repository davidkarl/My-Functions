function [ranges] = string_get_word_ranges_in_string(original_string)

words = strsplit(original_string);
counter=1;
for k=1:length(words)
    ranges(k,1) = counter;
    ranges(k,2) = counter+length(words{k});
    counter = counter + length(words{k});
end

