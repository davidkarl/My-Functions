function [new_string] = string_remove_ranges_from_string(original_string,ranges)

new_string = original_string;
script_string = 'new_string([';
for k=1:size(ranges,1)
    script_string = [script_string , 'ranges(' , num2str(k) , ',1):ranges(', num2str(k), ',2),']; 
end
script_string(end) = '';
script_string = [script_string, '])='''''];
eval(script_string);

