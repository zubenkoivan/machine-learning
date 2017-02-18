function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = sum(sum(((X * Theta') - Y) .^2 .* R)) / 2 + ...
    (sum(sum(Theta .* Theta)) + sum(sum(X .* X))) * lambda / 2;

X_grad = ((X * Theta') - Y) .* R * Theta;
Theta_grad = (((X * Theta') - Y) .* R)' * X;

grad = [X_grad(:); Theta_grad(:)] + lambda * params;

end
