% this function will try to split the node
% function [split,vars] = splitnode(x,impurity,classes)
%
% The input argument x is the data set for the node
% The input argumet impurity is the impurity for the node
% The input argument classes is the vector of class labels
% The output argument split will be the number to split on
% The output argument vars will be the dimension that is split on

function [split_value,split_dimension] = classification_tree_split_node(x,node_impurity,class_labels_vec,Nk,pies)

[n,dd] = size(x);
d = dd-1;
gos = -realmin;
split_value = [];
split_dimension = [];
for j =1:d  % check each variable and find the best split
   %[xs,ind]=sort(x(:,j));  
   xs = unique(x(:,j));		% get the unique values - sorted
   nxs = length(xs);
   for i = 1:(nxs-1)
      asplit = xs(i) + (xs(i+1)-xs(i))/2; %split bet points
      pl = (i)/n;
      pr = 1 - pl;
      % figure out how much is in one class or the other after the split
	  ileft = find(x(:,j) <= asplit);
	  iright = find(x(:,j) > asplit);
      for k=1:length(class_labels_vec)
         nktl(k) = length(find(x(ileft,dd)==class_labels_vec(k)));
         nktr(k) = length(find(x(iright,dd)==class_labels_vec(k)));
      end
      % get the new joint probabilities based on the split
      pcl = classification_tree_get_probability_of_node_t_and_class_j(Nk,pies,nktl);
      pcr = classification_tree_get_probability_of_node_t_and_class_j(Nk,pies,nktr);
	  newgos=node_impurity - pr*statistics_get_impurity_gini_index(pcr) - pl*statistics_get_impurity_gini_index(pcl);
	  if newgos > gos	% this is the best split so far
		  gos = newgos;
		  split_value = asplit;
		  split_dimension = j;
	  end
  end
end

% we have a matrix of all possible splits - asplit
% we have a matrix of the goodness of the splits - gos
% find the largest gos - largest decrease in impurity
% the function max should give the largest value for each
% column along with the row number that gave us that
%[mgos,indg]=max(gos);  % max goodness of split for each variable
%[msplit,dim]=max(mgos); % max gos over all variables
%split=asplit(indg(dim),dim);
