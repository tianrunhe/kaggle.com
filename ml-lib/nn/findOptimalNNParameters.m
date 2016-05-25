function lambda = findOptimalNNParameters(X, y, Xval, yval, input_layer_size, hidden_layer_size, num_labels)
	%returns the optimal lambda learning parameters to use for Neural Network

	lambda = 1.0;
	min_error = size(yval,1);

	for lambda_candidate=[0.03, 0.1, 0.3, 1, 3, 10]
		fprintf('\nTraining with lambda = %2.2f ...', lambda_candidate);
		nn_params = trainNN(X, y, input_layer_size, hidden_layer_size, num_labels, lambda_candidate);
		% Obtain Theta1 and Theta2 back from nn_params
		Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
	                 hidden_layer_size, (input_layer_size + 1));

		Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
	                 num_labels, (hidden_layer_size + 1));

		predictions = nnPredict(Theta1, Theta2, Xval);
		error = mean(double(predictions ~= yval));
		if error < min_error
			min_error = error;
			lambda = lambda_candidate;
		end
		fprintf('Error = %.2f. Best lambda = %2.2f\n', error, lambda);
	end

end
