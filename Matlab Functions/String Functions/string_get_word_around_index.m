function [word] = string_get_word_around_index(original_string,index)

if index==inf
    words = strsplit(original_string);
    word = words(end);
elseif index==1
    words = strsplit(original_string);
    word = words(1);
else
    
    ranges = string_return_word_ranges_in_string(original_string);
    if ~(isempty(find(ranges==index,1)))
        %if the index is one of the boundaries:
        word_number = find(ranges==index);
        word = string_return_word_number_from_string(original_string,word_number);
    else
        %if the index is within one of the boundaries:
        sorted_ranges = sort([ranges(:);index]);
        word_number = find(sorted_ranges==index)-1;
        word = string_return_word_number_from_string(original_string,word_number);
    end

end