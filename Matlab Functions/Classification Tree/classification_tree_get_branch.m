% function branch = getbranch(node,tree)
% this function will find the nodes to a branch so they can be pruned
% gets all of the children of the branch

function branch = classification_tree_get_branch(node,tree)

done = 0;
child = node;
i=1;
brnode{1} = child;
while ~done
   i=i+1;
   [done,brnode{i}] = classification_tree_get_children(brnode{i-1},tree);
end
branch = cat(1,brnode{:});
% get rid of the mother node - only send back the children of the node
branch(1)=[];
