clear all;clc;
figure;
data = load('ex2data2.txt');
x = data(:,[1,2]); y =data(:,3);
pos = find(y == 1);
neg = find(y == 0);
%%%%%
plot(x(pos,1), x(pos, 2), 'k+','MarkerSize', 8);
plot(x(neg,1), x(neg, 2), 'ro','MarkerSize', 8);
x = mapFeature(x(:,1), x(:,2));
theta = zeros(size(x,2),1);
%theta = 2*rand(size(x, 2), 1);
%theta = [1.273005
%  ;0.624876
%  ;1.177376
%  ;-2.020142
%  ;-0.912616
%  ;-1.429907  
%  ;0.125668
%  ;-0.368551
% ;-0.360033
%  ;-0.171068
%  ;-1.460894
%  ;-0.052499
%  ;-0.618889
%  ;-0.273745
%  ;-1.192301
% ;-0.240993
% ;-0.207934
%  ;-0.047224
%  ;-0.278327
%  ;-0.296602
%  ;-0.453957
%  ;-1.045511
%   ;0.026463
% ;-0.294330
%  ; 0.014381
%  ;-0.328703
%  ;-0.143796
%  ;-0.924883];
m = 118;lambda = 0.03;n = size(theta);
%theta = zeros(n+1,1);
alpha = 0.2;      %learning rate
%%%%%%%%%%%%%%%%%%%%%%
h = sigmoid(x*theta);
J = 1 / m * sum(-y.*log(h)-(1-y).*log(1-h));
printf('%f %f \n',J);
t = 1;
%%%%%%%%%%%%%%%%%%%%%%
do
%  for i = 1 : n
%   theta(i) = theta(i) - alpha*(1/m)* sum((h-y).*x(:,i))+lambda / m * theta(i);
%  end
  theta = theta - alpha*(1/m) * (x' * (h-y)) + lambda / m * theta;
  h = sigmoid(x*theta); 
  J = 1 / m * sum(-y.*log(h)-(1-y).*log(1-h))+ lambda / (2 * m) * sum(theta(2:end) .^ 2)
  t = t + 1;
 % plot(t,J,'bk');
 % hold on
until t == 8000 
%until J <= 0.452
printf('%f %f \n',J)
plotDecisionBoundary(theta, x, y);