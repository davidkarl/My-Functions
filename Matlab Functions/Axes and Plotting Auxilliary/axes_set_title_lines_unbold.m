function [] = axes_set_title_lines_unbold(lines_to_be_unbold)

if lines_to_be_unbold==inf
    %make all bold:
    title_strings = get(get(gca,'Title'),'String');
    for k=1:length(title_strings)
        title_strings{k} = strrep(title_strings{k},'\bf','');
        title_strings{k} = strrep(title_strings{k},'\rm','');
    end
    set(get(gca,'Title'),'String',title_strings)
else
    %according to specific input;
    title_strings = get(get(gca,'Title'),'String');
    for k=1:length(title_strings)
        if find(lines_to_be_unbold==k)
            title_strings{k} = strrep(title_strings{k},'\bf','');
            title_strings{k} = strrep(title_strings{k},'\rm','');
            title_strings{k} = [title_strings{k}];
        end
    end
    set(get(gca,'Title'),'String',title_strings)
end