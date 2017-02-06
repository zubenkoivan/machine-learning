function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

h = sigmoid(X * theta)';
thetaFrom2 = theta(2:end);
J = -(log(h) * y + log(1 - h) * (1 - y)) / m + lambda / (2 * m) * thetaFrom2' * thetaFrom2;
grad = ((h - y') * X / m)' + lambda / m * [0; thetaFrom2];

end
