function [clas,pclass,node] = classification_tree_classify_sample(x_sample,tree)
% CSTREEC     Classification label from a tree classifier.
%
%   [CLAS,PCLASS,NODE] = CSTREEC(X,TREE)
%   This function returns the class label and posterior probability
%   for a given unlabeled d-D feature vector X based on the classifier in
%   TREE. CLAS is the class label and PCLASS is the posterior
%   probability that X belongs to CLAS.  Note that PCLASS is a vector of
%   posterior probabilities. NODE is the terminal node for the observation X.
%
%   See also CSGROWC, CSPRUNEC, CSPLOTREEC, CSPICKTREEC

%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 

n = 1;	% start at node 1 - the root
term = 0;	% not a terminal node
while term ==0		% while not a terminal node
   % get split index and split value
   ind = tree.node(n).var;
   sval = tree.node(n).split;
   if x_sample(ind) <= sval		% go to the left
      n = tree.node(n).children(1);
   else
      n = tree.node(n).children(2);	% go to the right
   end
   term = tree.node(n).term;	% is this a terminal node
end
% once it is a terminal node, it drops out of the loop
% that will be the value of n
clas = tree.node(n).class;
pclass = tree.node(n).pclass;
node = n;