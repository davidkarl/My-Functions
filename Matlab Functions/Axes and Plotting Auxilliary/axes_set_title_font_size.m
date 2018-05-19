function [] = axes_set_title_font_size(font_size)

if length(font_size)==1
    set(get(gca,'Title'),'FontSize',font_size);
elseif length(font_size)>1
    title_strings = get(get(gca,'Title'),'String');
    for k=1:length(font_size)
        title_strings{k} = ['\fontsize{' num2str(font_size(k)) '}' title_strings{k}];
        set(get(gca,'Title'),'String',title_strings);
    end
end