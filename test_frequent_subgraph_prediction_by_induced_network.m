function [ net_predicted_full, precision, recall] = test_frequent_subgraph_prediction_by_induced_network( model, net_test, subgraphs, use_prctile, th, k_best )
%test_frequent_subgraph_prediction_by_induced_network This method builds a predicted network and evaluates its accuracy against a test network
%   This method uses a test network to induce a predicted network, on
%   observation of subgraph i in the test network. It uses the trained model
%   'model' to predict a lagged occurance of subgraph j. The test framework
%   then looks at the precision and recall of the predicted network

% @input model, a [S x S x 'horizon'] model trained from the train_frequent_subgraph_prediction_model method
% @input net_test, a [NxNxT] sequence (length T) of adjacency matrices used as a testing network
% @input subgraphs, a cell array of subgraphs used to evaluate 'model' on (generally the same as trained)
% @input use_prctile, a boolean for the prediction choice: 1: for percentile, otherwise: default to k best (default k=1)
% @input th [optional, use this or k_best], a scalar value determining the /percentile/ threshold to make predictions on j
% @input k_best [optional, default = 1, use this or th], a scalar value which selects the best (top-k) points on the time-subgraph probability surface given i.

% @output net_predicted, a [NxNxT] network induced by the predictions using the model
% @output precision, a scalar value reporting the precision on this evaluation
% @output recall, a scalar value reporting the recall on this evaluation

%example usage: test_frequent_subgraph_induce_using_prediction( model_trained, net_test, subgraphs_closed, 'prctile', 90);


%% default values, preprocessing
MODE_TH = 0;
MODE_TOPK = 1;

if(exist('th', 'var') && ~isempty(th)) %if th
    model(model <  prctile(model(model > 1), th)) = 0;
end

if(~exist('k_best', 'var') || isempty(k_best)) %if no k best
    k_best = 1;
end

if(exist('use_prctile', 'var') && use_prctile) %if selection mode is prctile
    selection_mode = MODE_TH;
else
    selection_mode = MODE_TOPK;
end

net_test(isnan(net_test)) = 0; % clean network of nans
[m,n,t] = size(net_test);
[s, ~, t_model] = size(model);

net_test = logical(reshape(net_test, m*n, t)); %reshape adjacency matrix to vector
net_predicted_full = false(m*n,t); %preallocate 
parfor(i = 1:s, matlabpool('size')) %for each subgraph
    net_predicted = false(m*n,t); %preallocate
    slice = squeeze(model(i,:,:));  %look at slice for subgraph i 
    if(selection_mode== MODE_TH) % use model prctile? 
        [sub_idx, lag_idx] = find(slice); %pre-thresholded above to avoid it in the loop
    elseif(selection_mode== MODE_TOPK)
        slice = slice(:);
        [weights, idx_sort] = sort(slice, 'descend');
        idx_kill = find(weights == 0, 1, 'first');
        [sub_idx, lag_idx]=ind2sub([s, t_model], idx_sort(1:min(k_best, idx_kill-1)));
    end    
    [sub_idx_unique, ~, t2_idx_unique] = unique(sub_idx); %get unique subgraphs predicted for i, t2_idx_unique gives the grouping where each unique item is associated. 
    test = all(net_test(subgraphs{i}, :), 1); %#ok<PFBNS>, find all indices where subgraph i occurs
    
    f = find(test)';
        
    if(~isempty(f))
        for j = 1:length(sub_idx_unique) %for each predicted j
            lags = lag_idx(t2_idx_unique == j); %find lags for this subgraph
            if(~isempty(lags))
                idx_add = unique(repmat(lags', length(f), 1) + repmat(f, 1, length(lags))); % matlab kungfu to build all predictions over occurances of i (f vector) with lags for this j
                idx_add(idx_add > t) = []; %exclude prediction indices greater than t
                net_predicted(subgraphs{sub_idx_unique(j)}, idx_add) = 1; %#ok<PFBNS>, make prediction
            end
        end
    end      
    net_predicted_full = net_predicted_full | net_predicted; %aggregate prediction 
end

precision = sum(sum(net_predicted_full & net_test))/sum(sum(net_predicted_full)); %calculate precision
recall = sum(sum(net_predicted_full & net_test))/sum(sum(net_test)); %calculate recall
net_predicted_full = reshape(net_predicted_full, m, n, t); %reshape to three dims
end

