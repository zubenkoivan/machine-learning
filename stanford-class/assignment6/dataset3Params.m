function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

CValues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmaValues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
errors = zeros(length(CValues) * length(sigmaValues), 1);
errorsSize = [length(CValues) length(sigmaValues)];

for i = 1:length(CValues)

    C = CValues(i);

    for j = 1:length(sigmaValues)

        sigma = sigmaValues(j);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        errors(sub2ind(errorsSize, i, j)) = mean(double(predictions ~= yval));

    end;

end;

[_ ind] = min(errors);
[i j] = ind2sub(errorsSize, ind);
C = CValues(i);
sigma = sigmaValues(j);

end
