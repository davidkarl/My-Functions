function [flag,relevant_cell_rows] = string_check_if_delimiter_is_in_string_or_string_array(original_string,delimiter)

counter=1;
if iscell(original_string)==1
   for k=1:length(original_string)
      flag(k) = ~(isempty(strfind(original_string{k},delimiter)));
      if flag(k)==1
         relevant_cell_rows(counter) = k;
         counter=counter+1;
      end
   end
else
    flag = ~(isempty(find(original_string,delimiter)));
end
