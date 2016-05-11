function digitRecognizer(trainingFileName, predictionFileName)

	%% Setup the parameters you will use for this exercise
	input_layer_size  = 784;  % 28x28 Input Images of Digits
	hidden_layer_size = 25;   % 25 hidden units
	num_labels = 10;          % 10 labels, from 0 to 9
	                          % (note that we have mapped "0" to label 1, "1" to label 2, etc)

	%% =========== Loading Data =============

	% Load Training Data
	fprintf('Reading Data from train.csv ...\n')
	X = csvread(trainingFileName);
	trainingExampleCount = int32((size(X, 1) - 1) * .7);

	% remove header
	X = X(2:end, :);
	% first column is the label, mapping 0-9 to 1-10
	y = X(:, 1) + 1;

	% remove the first column
	Xtrain = X(1:trainingExampleCount, 2:end);
	Xval = X(trainingExampleCount+1:end, 2:end);

	ytrain = y(1:trainingExampleCount,:);
	yval = y(trainingExampleCount+1:end,:);

	m = size(Xtrain, 1);

	%% =================== Draw Learning Curve ===================

	lambda = 1.0;
	fprintf('\nDrawing Learning Curve... \n')
	figure(1);
	[error_train, error_val] = ...
    	learningCurve(Xtrain, ytrain, Xval, yval, input_layer_size, hidden_layer_size, num_labels, lambda);
	plot(1:m, error_train, 1:m, error_val);

	title(sprintf('Neural Network Learning Curve (lambda = %f)', lambda));
	xlabel('Number of training examples')
	ylabel('Error')
	legend('Train', 'Cross Validation')


	%% =================== Training NN ===================

	fprintf('\nTraining Neural Network... \n')

	nn_params = trainNN(Xtrain, ytrain, input_layer_size, hidden_layer_size, num_labels, 1.5);

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