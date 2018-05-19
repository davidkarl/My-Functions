function [new_string] = string_substitute_range_with_delimiter(original_string,range,delimiter)

if range(1)==1
   str1 = '';
else
   str1 = original_string(1:range(1));
end

if range(2)==length(original_string)
   str2 = ''; 
else
   str2 = original_string(range(2):end);
end

new_string = strcat(str1,delimiter,str2);