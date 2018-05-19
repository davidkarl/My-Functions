function [] = axes_set_title_lines_bold(lines_to_be_bold)

if lines_to_be_bold==inf
    %make all bold:
    title_strings = get(get(gca,'Title'),'String');
    for k=1:length(title_strings)
        title_strings{k} = strrep(title_strings{k},'\rm','\bf');
    end
    set(get(gca,'Title'),'String',title_strings)
elseif lines_to_be_bold==0
    %make all normal:
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
        if find(lines_to_be_bold==k)
            title_strings{k} = strrep(title_strings{k},'\bf','');
            title_strings{k} = strrep(title_strings{k},'\rm','');
            title_strings{k} = ['\bf' title_strings{k} '\rm'];
        end
    end
    set(get(gca,'Title'),'String',title_strings)
end