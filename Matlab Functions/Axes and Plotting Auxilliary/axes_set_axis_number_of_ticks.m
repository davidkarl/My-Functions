function [] = axes_set_axis_number_of_ticks(flag_x_or_y,number_of_ticks)

if flag_x_or_y==1
   [x_limits] = xlim;
   x_min = x_limits(1);
   x_max = x_limits(2);
   new_x_tick = linspace(x_min,x_max,number_of_ticks);
   set(gca,'XTick',new_x_tick);
end
if flag_x_or_y==2
   [y_limits] = ylim;
   y_min = y_limits(1);
   y_max = y_limits(2);
   new_y_tick = linspace(y_min,y_max,number_of_ticks);
   set(gca,'XTick',new_y_tick);
end


