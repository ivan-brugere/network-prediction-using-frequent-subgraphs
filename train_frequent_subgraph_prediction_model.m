function [ model ] = train_frequent_subgraph_prediction_model( net_train, subgraphs, horizon, normalize )
%TRAIN_FREQUENT_SUBGRAPH_PREDICTION_MODEL Train a network prediction model
%using subgraphs

%This predictive model calculates the likelihood of 'lag' between pairwise occurances of subgraphs i and j. Intuitively, this method builds a distribution of: "if I observe i and not j,  at what lag will I observe j before observing i again?," the conditional probability of j and not i at time t+lag, given i and not j at time t.

% @input net_train, A [NxNxT] sequence (of length T) of adjacency matrices
% @input subgrap, A cell array where each element contains a vector of edges (e.g. indexes into the [NxN adjacency space]) , see: get_frequent_subgraphs.m
% @input horizon [optional, default: T], a scalar of the maximum horizon to search for pairwise lags
% @input normalize [optional, default: false], a boolean choice to normalize each subgraph pair i, j by the total occurances, e.g. return the probability distribution for the pair

% @output a likelihood matrix of size [S^2 - S x 'horizon] , S the number of subgraphs, each cut across time gives the observed lags of j after i (note this is nonsymmetric).  

% Citation: M. Lahiri and T.Y. Berger-Wolf. Structure Prediction in Temporal Networks using Frequent Subgraphs. Proc. IEEE CIDM 2007, Honolulu, Hawaii. April 2007. 


%% clean data, default values
net_train(isnan(net_train)) = 0;
[m,n,t] = size(net_train);

if(~exist('horizon', 'var') || isempty(horizon)) %if no horizon
    horizon = t; %use t
else
    horizon = min(horizon, t); %enforce max horizon
end

if(~exist('normalize', 'var') || isempty(normalize)) % if no normalize, dont normalize
    normalize = 0;
end

s = length(subgraphs);
net_train = logical(reshape(net_train, m*n, t)); %flatten adjacency matrix to 1D vector
model = zeros(s, s, horizon, 'single'); % preallocate

%% pairwise  
parfor(i = 1:s, matlabpool('size')) %for each subgraph
    model_surf = zeros(s, horizon, 'single'); %preallocate temp var for parfor slicing
    for k = 1:t-1 %for all time
        if(all(net_train(subgraphs{i}, k))) %#ok<PFBNS>, %if subgraph observed at time k
            id = setdiff(1:s, i); %remove self subgraph
            kill = find(all(net_train(subgraphs{i}, k+1:min(k+horizon, t)), 1), 1, 'first'); %find later observations of subgraph i
            if(~isempty(kill) && kill > 1) %if observed again              
                for j = id %for each non-self subgraph          
                    lag = find(all(net_train(subgraphs{j}, k+1:min(k+horizon, t)), 1)); %#ok<PFBNS>, find occurances
                    lag(lag >= kill) = []; %remove everything from when i is observed again
                    if(~isempty(lag)) %if observed js still exist
                        model_surf(j, lag) =  model_surf(j, lag) + 1;
                    end
                end
            end
        end        
    end
    ms = sum(model_surf, 2); % for normalization
    ms(~ms) = 1; %fix denom 
    if(normalize)
        model(i,:,:) = model_surf./repmat(ms, 1, horizon);
    else
        model(i,:,:) = model_surf;
    end
    disp(['Training: completed subgraph: ' num2str(i/s)]);
end
end

