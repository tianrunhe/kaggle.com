function digitRecognizer(trainingFileName, predictionFileName)

	%% Setup the parameters you will use for this exercise
	input_layer_size  = 784;  % 28x28 Input Images of Digits
	hidden_layer_size = 397;   % (28*28+10)/2 hidden units
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

	X = X(:, 2:end);

	m = size(Xtrain, 1);


	X_test = csvread(predictionFileName);
	% remove header
	X_test = X_test(2:end, :);

	% pred = nn(X, y, Xtrain, ytrain, Xval, yval, X_test, input_layer_size, hidden_layer_size, num_labels);
	pred = svm(X, y, Xtrain, ytrain, Xval, yval, X_test);
	pred = [[1:size(pred,1)]' pred-1];

	fprintf('\nSaving prediction to pred.csv...\n');
	filename = "pred.csv";
	FID = fopen(filename, 'w');
	fprintf(FID, "ImageId,Label\n");
	fclose(FID);
	dlmwrite(filename, pred, '-append', 'delimiter', ',');

end