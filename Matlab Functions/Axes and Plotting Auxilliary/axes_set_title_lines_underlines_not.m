function [] = axes_set_title_lines_underlines_not(lines_to_be_ununderline)

if lines_to_be_ununderline==inf
    %make all ununderlined:
    title_strings = get(get(gca,'Title'),'String');
    %clean all underlines:
    for k=1:length(title_strings)
        temp = title_strings{k};
        title_strings{k} = strrep(title_strings{k},'\underline{','');
        if strcmp(temp,title_strings{k})==1
            %there is not '\underline{...} so do nothing:
        else
            temp = title_strings{k};
            title_strings{k} = temp(1:end-1);
        end
    end
    %underline all:
    for k=1:length(title_strings)
        title_strings{k} = ['\underline{' title_strings{k} '}'];
    end
    set(get(gca,'Title'),'String',title_strings)
else
    %according to specific input;
    title_strings = get(get(gca,'Title'),'String');
    for k=1:length(title_strings)
        if find(lines_to_be_ununderline==k)
            temp = title_strings{k};
            title_strings{k} = strrep(title_strings{k},'\underline{','');
            if strcmp(temp,title_strings{k})==0
                %there was an '\underline{...}, chopp off closing '}':
                temp = title_strings{k};
                title_strings{k} = temp(1:end-1);
            else
                %there was no '\underline{...}, do nothing
            end
        end
    end
    set(get(gca,'Title'),'String',title_strings)
end