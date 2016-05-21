function svm(trainingFileName, predictionFileName)


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

	%% =========== Training =============
	
	% Try different SVM Parameters here
	%[C, sigma] = dataset3Params(Xtrain, ytrain, Xval, yval);
	C = 1.0;
	sigma  = 0.3;

	% Train the SVM
	model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));


	%% =========== Prediction =============
	X_test = csvread(predictionFileName);
	% remove header
	X_test = X_test(2:end, :);

	pred = svmPredict(model, X_test);
	% add row number as 1st column
	pred = [[1:size(pred,1)]' pred-1];

	fprintf('\nSaving prediction to pred.csv...\n');
	filename = 'pred.csv';
	FID = fopen(filename, 'w');
	fprintf(FID, "ImageId,Label\n");
	fclose(FID);
	dlmwrite(filename, pred, '-append', 'delimiter', ',');
end