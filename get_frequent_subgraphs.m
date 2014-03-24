function [ frequent, frequent_support, closed, closed_support ] = get_frequent_subgraphs( net, th,  slack, prct)
%GET_FREQUENT_SUBGRAPHS_PARALLEL Calculate frequent subgraphs on a dynamic
%network

% This method calculates the frequent subgraphs on a dynamic network using
% the apriori algorithm. Each level in the apriori lattace is independent
% and therefore calculated in parallel.

% A frequent subgraph is analogous to a frequent itemset with sufficient
% support and confidence. A frequent subgraph is a subgraph with sufficient
% support (occurances) over the time window. 

% This method returns frequent and closed subgraphs. A closed subgraph is one
% where no superset of the subgraph has equal support. We introduce a slack
% variable for when support is given over a large T (and exact equality is
% unlikely), in that case, |support(G1) - support(super(G1))| < slack is considered closed

% This method uses a triangular form (lower or upper) of the adjacency
% matrix (i.e. lower vs. upper is treated as directed). Using a full adjacency matrix will result in duplicate subgraphs and larger computation time. 

% @input net, A [NxNxT] sequence (of length T) of /triangular/ adjacency matrices
% @input th [optional, use this or prct], a scalar threshold value for support of a subgraph
% @input slack [optional], an error threshold allowing slack in the
% 'closed' subgraph definition. Used when T is large and the likelihood that the support of subgraph supersets is /exactly/ same is unreasonable
% @input prct [optional, use this or th], a scalar threshold value derived
% from the distribution of support for singleton subgraphs (e.g. single edges), example prct=90 builds subgraphs over the top-10% of edges by support.

% @output frequent, a cell array with edge IDs (indices into NxN)
% @output frequent_support, a verctor of support values for the above subgraphs
% @output closed,  a cell array (subset of 'frequent') with edge IDs (indices into NxN) where subgraphs satisfy the 'closed' property

%% default vars, clean data, fix thresholds
if(~exist('slack', 'var') || isempty(slack))
    slack = 0;
end

net(isnan(net)) = 0; %remove nans

[m,n,t] = size(net);

net = logical(reshape(net, m*n, t)); %collapse adjacency to 1D vector. This helps code efficiency/simplicity tremendously.  

closed_fail_map = containers.Map; %track when closure fails for a subgraph
support_map = containers.Map; %track support for each subgraph (map: O(1) access)
sup = single(nansum(net, 2)/t); %calculate support on singleton edges

if((~exist('th', 'var') || isempty(th)) && exist('prct', 'var') && ~isempty(prct))  %if no 'th' and 'prct'
    th = prctile(sup(sup > 0), prct); %drawing percentile on support > 0 to ignore likely many 0 values since we are using lower-tri of matrix. 
end

%% find k = 1 size frequent subgraphs (singleton edges)
idx = find(sup > th); %get singleton subgraphs
candidates = cell(length(idx), 1);
for i = 1:length(idx) %for each singleton
    candidates{i} = idx(i); %build candidates list
    support_map(num2str(idx(i))) = sup(idx(i)); %fill support map
end
clear idx sup;

k = 2; %build subgraphs of size 2 (recall: 2 edges)
while length(candidates) >= 2

    %% build all combinations from subgraph candidates i, j
    candidate_idx = 1;
    curr_idx = 1;
    combination_graphs_list = cell(nchoosek(length(candidates), 2), 1); %preallocate based on number of pairs of subgraphs
    idx = zeros(nchoosek(length(candidates), 2), 2, 'single'); %preallocate index into candidate list ('candidates')
    subset_len = 0; %used for later preallocation
    for i = 1:length(candidates) %for each subgraph
        for j = i+1:length(candidates) %for each subgraph after i
            combination_graphs_list{curr_idx} = nchoosek(unique([candidates{i}, candidates{j}]), k); %enumerate ways to combine edges from candidates i, j into subsets of size k
            subset_len = size(combination_graphs_list{curr_idx},1)+subset_len; %keep track of length
            idx(curr_idx, :) = [i, j]; %track indices of candidate combinations
            curr_idx = curr_idx+1; 
        end
    end
    %key: 'combination_graphs_list' is a cell vector containing the enumeration of all larger graphs from the union of the edges between i and j. 
    
    %% preallocate new candidates and support  
    candidates_new = cell(subset_len, 1); %preallocate based on number of possible subsets
    support_cell = cell(length(combination_graphs_list), 1); %preallocate support 
    
    disp(['Number of subsets of size ' num2str(k) ': ' num2str(subset_len)])
    
    %% do hard calculation: query on graph for subgraph support (map)
    parfor (i = 1:length(combination_graphs_list), matlabpool('size')) %for each subset
        combination_graphs = combination_graphs_list{i}; %get some ij candidate list from above
        sup = zeros(size(combination_graphs, 1), 1, 'single'); %preallocate support
        for l = 1:size(combination_graphs, 1) %for each combined graph
            sup(l) = single(sum(all(net(combination_graphs(l, :), :), 1))/t); %calculate support
        end
        support_cell{i} = sup; 
    end
    
    %% build new candidate list from calculated support (reduce)  
    for i = 1:length(support_cell) %for each combination 
        sup = support_cell{i}; %get support vector
        combination_graphs = combination_graphs_list{i};
        
        for l = 1:size(combination_graphs, 1) %for each possible new candidate 
            if(sup(l) > th) %support passes th? 
                candidates_new{candidate_idx} = combination_graphs(l, :); %add to new candidates list
                candidate_idx = candidate_idx+1;
                
                %% build text index for maps
                str_idx = strrep(num2str(combination_graphs(l, :)), '  ', ' '); %space delimited new candidate
                support_map(str_idx) = sup(l);
                super1 = strrep(num2str(candidates{idx(i, 1)}), '  ', ' '); %using idx to recover combined edgelist i, j
                super2 = strrep(num2str(candidates{idx(i, 2)}), '  ', ' '); %using idx to recover combined edgelist i, j
                
                if(abs(sup(l) - support_map(super1)) <= slack) %superset fails closed test?
                    closed_fail_map(super1) = 1;
                end
                if(abs(sup(l) - support_map(super2)) <= slack)
                    closed_fail_map(super2) = 1;
                end
            end
        end
    end
    k = k +1;
    candidates_new = candidates_new(~cellfun(@isempty,candidates_new)); %build new candidates list
    candidates = candidates_new;
end
key = keys(support_map); %get keys
[keys_closed, keys_closed_idx] = setdiff(key, keys(closed_fail_map)); %closed is everything that didnt fail
frequent = cellfun(@str2num, key , 'UniformOutput', false); %parses edgelist ids back to numeric
clear key;

closed = cellfun(@str2num, keys_closed , 'UniformOutput', false); %parses edgelist ids back to numeric
clear keys_closed;

frequent_support = single(cell2mat(values(support_map))); %build support vector
closed_support = frequent_support(keys_closed_idx); %build support vector
end