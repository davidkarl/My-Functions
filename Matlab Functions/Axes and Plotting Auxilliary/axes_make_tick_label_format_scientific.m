function [] = axes_make_tick_label_format_scientific(flag_x_or_y,number_of_digits,number_of_ticks,varargin)

if flag_x_or_y==1
   if nargin==2
       tick_labels = get(gca,'XTick');
       new_tick_labels = scientific(tick_labels,number_of_digits);
       set(gca,'XTickLabel',char(new_tick_labels));
   elseif nargin==3
       x_limits = xlim;   
       x_min = x_limits(1);
       x_max = x_limits(2);
       new_ticks = linspace(x_min,x_max,number_of_ticks);
       new_tick_labels = scientific(new_ticks,number_of_digits);
       set(gca,'XTick',new_ticks);
       set(gca,'XTickLabel',char(new_tick_labels'));
   end   
end   
if flag_x_or_y==2 
   if nargin==2 
       tick_labels = get(gca,'YTick');
       new_tick_labels = scientific(tick_labels,number_of_digits);
       set(gca,'YTickLabel',char(new_tick_labels));
   elseif nargin==3
       y_limits = ylim; 
       y_min = y_limits(1);
       y_max = y_limits(2);
       new_ticks = linspace(y_min,y_max,number_of_ticks);
       new_tick_labels = scientific(new_ticks,number_of_digits);
       set(gca,'YTickLabel',char(new_tick_labels));
   end
end
