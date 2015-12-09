function[J,g] = costfunction(theta)
data = load('ex2data1.txt');
x = data(:,[1,2]); y =data(:,3);theta = zeros(2,1);
[m, n] = size(x);
X = [ones(m,1),x];
h = sigmoid(X*theta);
J = 1 / m * sum(-y.*log(h)-(1-y).*log(1-h));