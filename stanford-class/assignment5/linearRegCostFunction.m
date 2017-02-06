function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
error = X * theta - y;
thetaFrom2 = theta(2:end);
J = (error' * error + thetaFrom2' * thetaFrom2 * lambda) / (2 * m);
grad = ((error' * X)' + [0; thetaFrom2] * lambda) / m;

end
