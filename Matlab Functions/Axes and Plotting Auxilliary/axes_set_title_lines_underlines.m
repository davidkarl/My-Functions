function [] = axes_set_title_lines_underlines(lines_to_be_underline)

if lines_to_be_underline==inf
    %make all underlined for a fresh start:
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
elseif lines_to_be_underline==0
    %make all normal (dont yet know exactly how, i'll go arabic) and:
    %i assume each line is either completely underlined or isn't
    title_strings = get(get(gca,'Title'),'String');
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
    set(get(gca,'Title'),'String',title_strings)
else
    %according to specific input;
    title_strings = get(get(gca,'Title'),'String');
    for k=1:length(title_strings)
        if find(lines_to_be_underline==k)
            temp = title_strings{k};
            title_strings{k} = strrep(title_strings{k},'\underline{','');
            if strcmp(temp,title_strings{k})==0
                %there was an '\underline{...} so do nothing
            else
                title_strings{k} = ['\underline{' title_strings{k} '}'];
            end
        end
    end
    set(get(gca,'Title'),'String',title_strings)
end