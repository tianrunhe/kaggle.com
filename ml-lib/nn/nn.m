function pred = nn(X, y, Xtrain, ytrain, Xval, yval, X_test, input_layer_size, hidden_layer_size, num_labels)

	%% =========== Training =============
	
	% Try different lambda here
	lambda = findOptimalNNParameters(Xtrain, ytrain, Xval, yval, input_layer_size, hidden_layer_size, num_labels);

	% Train NN with best lambda
	nn_params = trainNN(X, y, input_layer_size, hidden_layer_size, num_labels, lambda);
	% Obtain Theta1 and Theta2 back from nn_params
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


	%% =========== Prediction =============
	pred = nnPredict(Theta1, Theta2, Xval);

end