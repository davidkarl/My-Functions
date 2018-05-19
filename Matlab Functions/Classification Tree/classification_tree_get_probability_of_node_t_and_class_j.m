% function pc = getprob(Nk,pies,Nkt)
%
% Nk is a vector containing the number of cases in the learning sample
% that belong to each class.
% pies is a vector containing the prior probabilities that it belongs to a class
% Nkt is a vector of the number of cases in the node t that belong to each class

function pc = classification_tree_get_probability_of_node_t_and_class_j(number_of_samples_per_class,pies_vec,number_of_samples_per_class_in_node_t)

pkt = pies_vec.*number_of_samples_per_class_in_node_t./number_of_samples_per_class;
pt = sum(pkt);
pc = pkt/pt;

