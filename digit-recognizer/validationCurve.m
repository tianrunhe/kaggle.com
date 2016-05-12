function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval, input_layer_size, hidden_layer_size, num_labels)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval, input_layer_size, hidden_layer_size, num_labels) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
	lambda = lambda_vec(i);
	fprintf('\nValidation Curve with lambda = %.2f\n', lambda)
	[nn_params] = trainNN(X, y, input_layer_size, hidden_layer_size, num_labels, lambda);

	[error_train(i), temp] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, zeros(size(lambda)));
	[error_val(i), temp] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, Xval, yval, zeros(size(lambda)));
endfor


end
