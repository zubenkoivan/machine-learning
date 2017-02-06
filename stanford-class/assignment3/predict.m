function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
X = [ones(1, m); X'];
a2 = sigmoid(Theta1 * X);
a2 = [ones(1, m); a2];
a3 = sigmoid(Theta2 * a2);
[_ indices] = max(a3);
p = indices';

end
