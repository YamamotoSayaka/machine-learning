data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

figure; % a new figure window
plot(X, y, 'rx', 'MarkerSize', 10); % data 

x = [ones(m, 1), data(:,1)];
theta = zeros(1,2);
alpha = 0.01;
h = x * theta';
j = 1 / (2*m)*sum((h-y).^2);
do
    theta(1) = theta(1) - alpha * (1/m) * sum(h-y);
    theta(2) = theta(2) - alpha * (1/m) * sum((h - y) .* X);
    theta = [theta(1), theta(2)];
    h = x * theta';
    j = 1 / (2*m)*sum((h-y).^2)  %costfunction
until (j <= 4)   
fprintf('%f %f \n', theta(1), theta(2));
    