function [] = axes_set_legend_font_size(font_size)

children_gcf = get(gcf,'children');
legend_handle = children_gcf(2);
set(legend_handle,'FontSize',font_size);

