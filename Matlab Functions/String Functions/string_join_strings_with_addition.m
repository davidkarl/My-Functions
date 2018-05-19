function [new_string] = string_join_strings_with_addition(original_strings,additions_between_strings)

if ~isempty(original_strings)
    n = numel(original_strings);
    if n > 1        
        % augment the elements with delimiter
        if ~isempty(delimiter)
            new_strings = [cellfun(@(x) [x, additions_between_strings], original_strings(1:end-1), 'UniformOutput', false), ...
                original_strings(end)];            
        else
            new_strings = original_strings;
        end
        
        % concatenate them 
        s = [original_strings{:}];
    else
        s = original_strings{1};
    end
else
    s = '';
end


