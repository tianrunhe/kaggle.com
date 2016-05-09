function digitRecognizer(trainingFileName, predictionFileName, maxIter, lambda)

	%% Setup the parameters you will use for this exercise
	input_layer_size  = 784;  % 28x28 Input Images of Digits
	hidden_layer_size = 25;   % 25 hidden units
	num_labels = 10;          % 10 labels, from 0 to 9
	                          % (note that we have mapped "0" to label 1, "1" to label 2, etc)

	%% =========== Loading Data =============

	% Load Training Data
	fprintf('Reading Data from train.csv ...\n')
	X = csvread(trainingFileName);

	% remove header
	X = X(2:end, :);

	% first column is the label, mapping 0-9 to 1-10
	y = X(:, 1) + 1;

	% remove the first column
	X = X(:, 2:end);

	m = size(X, 1);


	%% ================ Initializing Pameters ================

	fprintf('\nInitializing Neural Network Parameters ...\n')

	initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
	initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

	% Unroll parameters
	initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


	%% =================== Training NN ===================

	fprintf('\nTraining Neural Network... \n')

	options = optimset('MaxIter', maxIter);

	% Create "short hand" for the cost function to be minimized
	costFunction = @(p) nnCostFunction(p, ...
	                                   input_layer_size, ...
	                                   hidden_layer_size, ...
	                                   num_labels, X, y, lambda);

	% Now, costFunction is a function that takes in only one argument (the
	% neural network parameters)
	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

	% Obtain Theta1 and Theta2 back from nn_params
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
	                 hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
	                 num_labels, (hidden_layer_size + 1));


	%% ================= Prediction =================

	X_test = csvread(predictionFileName);
	% remove header
	X_test = X_test(2:end, :);

	pred = predict(Theta1, Theta2, X_test);
	% add row number as 1st column
	pred = [[1:size(pred,1)]' pred-1];

	fprintf('\nSaving prediction to pred.csv...\n');
	filename = "pred.csv";
	FID = fopen(filename, 'w');
	fprintf(FID, "ImageId,Label\n");
	fclose(FID);
	dlmwrite(filename, pred, '-append', 'delimiter', ',');

end