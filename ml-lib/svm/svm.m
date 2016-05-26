function pred = svm(X, y, Xtrain, ytrain, Xval, yval, X_test)

	%% =========== Training =============
	
	% Try different SVM Parameters here
	[C, sigma] = findOptimalSVMParametersForGaussianKernel(Xtrain, ytrain, Xval, yval);

	% Train the SVM
	model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));


	%% =========== Prediction =============

	pred = svmPredict(model, X_test);
	% add row number as 1st column

end