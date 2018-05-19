function classification_tree_plot_tree(tree)
% CSPLOTREEC    Plot a classification tree.
%
%   CSPLOTREEC(TREE)
%   This function plots a classification tree given by TREE.
%
%   See also CSGROWC, CSPRUNEC, CSTREEC, CSPICKTREEC

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 

% constants for plotting
% change this for each level
rcal = inline('1/sin(theta*pi/180)');
dex = inline('sqrt(r^2-1)');
offset = 0.04;
theta = 55;
r = rcal(theta);
delx = dex(r);

% plot root node to get started
y = -1;
x = 0;
tree.node(1).y=-1;
tree.node(1).x=0;
plot(x,y,'o')
text(x+offset,y,['x',int2str(tree.node(1).var), ' < ', num2str(tree.node(1).split,'%0.2g')])
hold on

% now plot each level.
oldchild = 1;
[done,newchild] = classification_tree_get_children(oldchild,tree);	% get the children at the next level
done=0;
while ~done
   [done,newchild] = classification_tree_get_children(oldchild,tree);	% get the children at the next level
   newchild = sort(newchild);
   theta = theta+5;
   r = rcal(theta);
   delx = dex(r);
   % loop through all of the nodes and connect parent to child
   for i = 1:length(newchild)  % occur in pairs
      node = newchild(i);
      parent = tree.node(node).parent;
      y = tree.node(parent).y;
      x = tree.node(parent).x;
      yc = y-1;
      if mod(node,2) == 0	% it is a left child
         xc = x-delx;
      else
         xc = x+delx;
      end
      plot([x xc],[y yc])
      plot(xc,yc,'o')
      drawnow
      % if the nodes have children, then plot the split, else show the class label
      if tree.node(node).term == 0		% not a terminal node
         text(xc+offset,yc+offset,...
            ['x',int2str(tree.node(node).var), ' < ', num2str(tree.node(node).split,'%0.2g')])
         % set the x,y coordinates for the nodes
         tree.node(node).x = xc;
         tree.node(node).y = yc;
      else 
		 text(xc-offset,yc-3*offset,['C- ',int2str(tree.node(node).class)])
      end
   end   % end for loop
   oldchild = newchild;
end
hold off
ax= axis;
axis([ax(1)-.1 ax(2)+.1 ax(3)-.1 ax(4) + .1])
axis off


      
      
   


