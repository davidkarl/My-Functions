function [] = axes_set_axis_label_font_size(flag_x_or_y,font_size)

if flag_x_or_y==1 || flag_x_or_y==3
   set(get(gca,'XLabel'),'FontSize',font_size); 
end
if flag_x_or_y==2 || flag_x_or_y==3
   set(get(gca,'YLabel'),'FontSize',font_size);
end

