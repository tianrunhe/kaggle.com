function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, input_layer_size, hidden_layer_size, num_labels, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%

m = size(X, 1);
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i  = 1:m
	fprintf('\nComputing learning curve for %d examples\n', i)
	[nn_params] = trainNN(X(1:i,:), y(1:i), input_layer_size, hidden_layer_size, num_labels, lambda);
	[error_train(i), temp] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X(1:i, :), y(1:i), zeros(size(lambda)));
	fprintf('\nTraing Error is %.2f\n', error_train(i))
	[error_val(i), temp] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, Xval, yval, zeros(size(lambda)));
	fprintf('\nValidation Error is %.2f\n', error_val(i))
endfor

end
