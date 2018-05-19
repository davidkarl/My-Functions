function [] = axes_set_axis_tick_font_size(font_size)

%i cant only change x axis or y axis so i have to do both:
set(gca,'XTickLabel',get(gca,'XTickLabel'),'FontSize',font_size);
set(gca,'XTickLabelMode','auto');
