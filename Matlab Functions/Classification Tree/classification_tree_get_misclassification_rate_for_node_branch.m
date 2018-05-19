% function gt = brnchmisclas(node,tree)
% this function will find the misclassification rate for a branch
% that starts at 'node' Actually, this will find the following function
% as described in Breiman, et al. 
% g(t)=(R(t)-R(Tt))/(numterm(Tt)-1);

function gt = classification_tree_get_misclassification_rate_for_node_branch(node,tree)

[term,misclass,pt] = getinfo(tree); % these are for the whole tree
% get all of the children of the branch node
done = 0;
i=1;
brnode{1} = node;
while ~done
   i=i+1;
   [done,brnode{i}] = classification_tree_get_children(brnode{i-1},tree);%this gets all nodes at a level of the tree
end
brn = cat(1,brnode{:});  % gives a vector of nodes that are in a branch 
%this yields the information for all nodes belonging to the branch:
termb = term(brn);  % get the values that belong to these nodes
misclassb = misclass(brn);
ptb = pt(brn);
%of these nodes in the branch, find the ones that are terminal:
indt = find(termb==1);
numterm = length(indt);  % this is the number of terminal nodes in branch Tt
RTt = sum(misclassb(indt).*ptb(indt));  % misclassifcation for the branch
% get the information for the parent of the branch
Rt = tree.node(node).misclass*tree.node(node).pt;
% calculate the misclassification for a branch
gt = (Rt-RTt)/(numterm-1);

