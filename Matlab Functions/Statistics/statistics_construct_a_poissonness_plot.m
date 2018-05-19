function statistics_construct_a_poissonness_plot(bin_axis_vec, counts_vec)
% CSPOISSPLOT Construct a Poissonness plot.
%
%   CSPOISSPLOT(K, NK) Constructs a Poissonness plot, which
%   is used to graphically determine whether the observed
%   counts follow a Poisson distribution. The inputs to the
%   function are a vector of counts K and the frequency of
%   occurrence NK.
%
%   EXAMPLE:
%
%   k = 0:6;
%   n_k = [156 63 29 8 4 1 1];
%   cspoissplot(k,n_k)
%
%   See also CSPOISP, CSPOISC, CSPOIRND, CSPOIPAR, CSBINOPLOT


%   W. L. and A. R. Martinez, 9/15/01
%   Computational Statistics Toolbox 

bin_axis_vec = 0:6;
counts_vec = [156,63,29,8,4,1,1];

%poissoness plot - basic:
N = sum(counts_vec);

%get vector of factorials:
factrial_vec = zeros(size(bin_axis_vec));
for i = bin_axis_vec
   factrial_vec(i+1) = factorial(i);
end

%get phi(n_k) for plotting:
phik = log( factrial_vec .* counts_vec/sum(counts_vec) );

%find the counts that are equal to 1 plot these with the symbol 1 plot rest with a symbol:
ind = find(counts_vec~=1);
plot(bin_axis_vec(ind),phik(ind),'o')
ind = find(counts_vec==1);
if ~isempty(ind)
   text(bin_axis_vec(ind),phik(ind),'1')
end
% add some whitespace to see better
axis([-0.5 max(bin_axis_vec)+1 min(phik)-1 max(phik)+1])
xlabel('Number of Occurrences - k')
ylabel('\phi (n_k)')
