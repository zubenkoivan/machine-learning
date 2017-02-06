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

XX = [ones(m, 1) X];
yy = zeros(m, num_labels);
yy(sub2ind(size(yy), 1:m, y')) = 1;

z2 = XX * Theta1';
a2 = [ones(m, 1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

J = -sum(sum(log(a3) .* yy + log(1 - a3) .* (1 - yy))) / m;
R = (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2))) * lambda / (2 * m);
J = J + R;

delta3 = a3 - yy;

Theta2_grad = kron(a2,ones(size(delta3, 2), 1)) .* delta3'(:);
Theta2_grad = reshape(Theta2_grad, size(delta3, 2), m, size(a2, 2));
Theta2_grad = reshape(sum(Theta2_grad, 2) / m, size(delta3, 2), size(a2, 2));
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + Theta2(:, 2:end) * lambda / m;

delta2 = delta3 * Theta2(:, 2:end) .* sigmoidGradient(z2);

Theta1_grad = kron(XX,ones(size(delta2, 2), 1)) .* delta2'(:);
Theta1_grad = reshape(Theta1_grad, size(delta2, 2), m, size(XX, 2));
Theta1_grad = reshape(sum(Theta1_grad, 2) / m, size(delta2, 2), size(XX, 2));
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + Theta1(:, 2:end) * lambda / m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
