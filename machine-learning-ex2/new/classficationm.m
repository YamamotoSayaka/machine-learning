clear all;clc;figure;
data = load('ex2data2.txt');
x = data(:,[1,2]); y =data(:,3);A = [];
pos = find(y == 1);neg = find(y == 0);
%%%%%%%%%%%%%%%%%%%%
plot(x(pos,1), x(pos, 2), 'k+','MarkerSize', 8);
plot(x(neg,1), x(neg, 2), 'ro','MarkerSize', 8);
hold on;
x = mapFeature(x(:,1), x(:,2));
theta = zeros(size(x,2),1);
%theta = 2*rand(size(x, 2), 1);
m = 118;lambda = 0.03;n = size(theta);
alpha = 0.06666;      %learning rate
%%%%%%%%%%%%%%%%%%%%%%
h = sigmoid(x*theta);
J = 1 / m * sum(-y.*log(h)-(1-y).*log(1-h));
%printf('%f %f \n',J);
t = 1;A(t) = J;
%%%%%%%%%%%%%%%%%%%%%%
do
%  for i = 1 : n
%   theta(i) = theta(i) - alpha*(1/m)* sum((h-y).*x(:,i))+lambda / m * theta(i);
%  end
  theta = theta - alpha*(1/m) * (x' * (h-y)) + lambda / m * theta;
  h = sigmoid(x*theta); 
  J = 1 / m * sum(-y.*log(h)-(1-y).*log(1-h))+ lambda / (2 * m) * sum(theta(2:end) .^ 2);
  t = t + 1;
  A(t) = J;
until t == 5000 %%min J later through the figure
%until J <= 0.6
printf('%f %f \n',J)
plotDecisionBoundary(theta, x, y);
hold off;
figure;plot(A,'bk');