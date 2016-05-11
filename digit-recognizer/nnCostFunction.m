function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1) X];

Y = zeros(m, num_labels);
for i = 1:m
	Y(i, y(i, 1)) = 1;
endfor

A_one = X; # 5000 * 401
Z_two = A_one * Theta1'; # 5000 * 401 * 401 * 25
A_two = sigmoid(Z_two); # 5000 * 25
A_two = [ones(size(A_two, 1),1) A_two]; # 5000 * 26
Z_three = A_two * Theta2'; # 5000 * 26 * 26 * 10
A_three = sigmoid(Z_three); # 5000 * 10
h = A_three;

for i = 1:m
	for k = 1:num_labels
		J += -Y(i,k) * log(h(i,k)) - (1-Y(i,k))*log(1-h(i,k));
	endfor
endfor
J = J ./ m;

temp = Theta1; # 25 * 401
temp(:, 1) = zeros(size(Theta1, 1), 1);
J += lambda / (2 * m) * sumsq(temp(:));
temp = Theta2;
temp(:, 1) = zeros(size(Theta2, 1), 1);
J += lambda / (2 * m) * sumsq(temp(:));

delta_1 = zeros(size(Theta1));
delta_2 = zeros(size(Theta2));
for t = 1:m
	# Step 1, Feedforward
	a_1 = X(t,:); # 1 * 401
	z_2 = a_1 * Theta1'; # 1 * 401 * 401 * 25
	a_2 = sigmoid(z_2); # 1 * 25
	a_2 = [ones(size(a_2, 1), 1) a_2]; # 1 * 26
	z_3 = a_2 * Theta2'; # 1 * 26 * 26 * 10
	a_3 = sigmoid(z_3); # 1 * 10
	# Step 2
	error_3 = (a_3 - Y(t, :))'; # 10 * 1
	# Step 3
	error_2 = Theta2' * error_3 .* [ 1 ; sigmoidGradient(z_2')]; # 26*10 * 10*1 .* 26*1 -> 26*1
	delta_1 += error_2(2:end) * a_1; # 25*1 * 1*401 -> 25*401
	delta_2 += error_3 * a_2; # 10*1 * 1*26 -> 10*26

	temp = 1 ./ m * delta_1; # 25*401
	temp(:, 2:end) += lambda ./ m * Theta1(:, 2:end);
	Theta1_grad = temp;
	temp = 1 ./ m * delta_2; # 10 * 26
	temp(:, 2:end) += lambda ./ m * Theta2(:, 2:end);
	Theta2_grad = temp;
endfor

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
