function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
h = sigmoid((X * theta)');
thetaFrom2 = theta(2:end);
J = -(log(h) * y + log(1 - h) * (1 - y)) / m + ...
     thetaFrom2' * thetaFrom2 * lambda / (2 * m);
grad = (((h - y') * X)' + [0; lambda * theta(2:end)]) / m;

end
