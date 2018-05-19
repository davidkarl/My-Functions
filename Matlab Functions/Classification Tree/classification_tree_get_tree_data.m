% function [term,nkt,imp] = getdata(tree)
% This function will extract the data needed from the tree. These data
% are the vector of terminal nodes, the number of points in each node
% and the impurity for each node.

function [terminal_nodes,number_of_points_in_each_node,impurity_index_in_each_node] = ...
                                                                classification_tree_get_tree_data(tree)

tp = squeeze(struct2cell(getfield(tree,'node')));
terminal_nodes = cat(1,tp{1,:});  % this should get the terminal node flags
number_of_points_in_each_node = cat(1,tp{2,:});	% this should get the number of points in each node
impurity_index_in_each_node = cat(1,tp{3,:});	% this should get the impurity of the nodes


