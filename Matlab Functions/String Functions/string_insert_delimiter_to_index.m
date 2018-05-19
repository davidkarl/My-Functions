function [new_string] = string_insert_delimiter_to_index(original_string,index,delimiter)

if index==1
    new_string = strcat(delimiter,original_string);
elseif index==length(original_string)
    new_string = strcat(original_string,delimiter); 
else
    str1 = original_string(1:range(1));
    str2 = original_string(range(1)+1:end);
    new_string = strcat(str1,delimiter,str2);
end




