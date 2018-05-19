% function gt = brnchmisclas(node,tree)
% this function will find the misclassification rate for a branch
% that starts at 'node' Actually, this will find the following function
% as described in Breiman, et al. 
% g(t)=(R(t)-R(Tt))/(numterm-1);

function gt = classification_tree_prune_branch(node,tree)

numterm = length(tree.termnodes);
[term,misclass] = getinfo(tree); %Matlab function
brmis = 0;
% get all of the children of the branch node
done = 0;
child = node;
i = 1;
brnode{1} = child;
while ~done
   i = i+1;
   [done,brnode{i}] = classification_tree_get_children(brnode{i-1},tree);
end
brn = cat(1,brnode{:});
for i=1:length(brn)
   ind = brn(i);
   if term(ind) ==1 	% if it is terminal
      brmis = brmis + misclass(ind);
   end
end
% Note:  brmis = R(Tt)
gt = (misclass(node)-brmis)/(numterm-1);
