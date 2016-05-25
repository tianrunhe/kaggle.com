function [C, sigma] = findOptimalSVMParametersForGaussianKernel(X, y, Xval, yval)
%returns the optimal (C, sigma) learning parameters to use for SVM
%with gaussian kernel

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

min_error = size(yval,1);

for C_candidate=[0.03, 0.1, 0.3, 1, 3, 10]
	for sigma_candidate=[0.03, 0.1, 0.3, 1, 3, 10]
		fprintf('\nTraining with C = %2.2f, sigma = %2.2f ...', C_candidate, sigma_candidate);
		model = svmTrain(X, y, C_candidate, @(x1, x2) gaussianKernel(x1, x2, sigma_candidate)); 
		predictions = svmPredict(model, Xval);
		error = mean(double(predictions ~= yval));
		if error < min_error
			min_error = error;
			C = C_candidate;
			sigma = sigma_candidate;
		end
		fprintf('Error = %.2f. Best parameters so far are: C = %2.2f, sigma = %2.2f\n', error, C, sigma);
	end
end



% =========================================================================

end
