This project is an implementation of (Lahiri & Berger-Wolf 2007), "Structure Prediction in Temporal Networks using Frequent Subgraphs" for MATLAB. 

The method extracts all subgraphs at a user-specified "support" threshold, and calculates the probability density map of subgraph 'j' following observed subgraph 'i' before the next occurrence of 'i.' This is the conditional probability of observing 'j' at a 'time lag' given the observation of 'i'.

Observing 'i' on testing data, the model uses the maximum likelihood of possible 'j' subgraphs at all lags to induce a predicted network.

Example script:

support = 70; %adaptive threshold from 70th percentile and above of singleton subgraph (i.e. single edge)
th_topk = 2;
[ ~, ~, closed, ~ ] = get_frequent_subgraphs( net_test, [], 1e-3, support);  % get frequent subgraphs on net_test with support > 70 percentile
[ model_trained ] = train_frequent_subgraph_prediction_model(  network_train, closed, 6000 ); %train network prediction model at time horizon 6000 steps.
[ ~, precision, recall] = test_frequent_subgraph_prediction_by_induced_network(model_train, network_test, closed, [], [], th_topk(k));
disp(num2str(precision));
disp(num2str(recall));

Citation: 

M. Lahiri and T.Y. Berger-Wolf. Structure Prediction in Temporal Networks using Frequent Subgraphs. Proc. IEEE CIDM 2007, Honolulu, Hawaii. April 2007. 

Contact:

Ivan Brugere
Computational Populations Biology Lab
University of Illinois at Chicago
ibruge2@uic.edu