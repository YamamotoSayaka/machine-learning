figure;
data = load('ex2data1.txt');
x = data(:,[1,2]); y =data(:,3);
[m, n] = size(x);
X = [ones(m,1),x];
%theta = 30.*rand(3,1);
%theta = zeros(n+1,1);
theta = [-24;0.4;0.4];
alpha = 0.0009999;      %learning rate
%%%%%%%%%%%%%%%%%%%%%%
h = sigmoid(X*theta);
J = 1 / m * sum(-y.*log(h)-(1-y).*log(1-h)); %h会取到 0 1精度问题
t = 1;
%%%%%%%%%%%%%%%%%%%%%%
printf('%f %f \n',theta)
do
  for i = 1 : ( n+1 )
  theta(i) = theta(i) - alpha*(1/m)* sum((h-y).*X(:,i));
  end
  %theta(1) = theta(1) - alpha*(1/m)* sum((h-y).*X(:,1));
  %theta(2) = theta(2) + alpha*(1/m)* sum((h-y).*X(:,2));
  %theta(3) = theta(3) + alpha*(1/m)* sum((h-y).*X(:,3));
  h = sigmoid(X*theta);  %%%sigmoid function
  J = 1 / m * sum(-y.*log(h)-(1-y).*log(1-h))
  plot(t,J,'bk');
  hold on
  t = t + 1;
until t == 3000 %%%由fminunc计算得的costfunction为0.233498
printf('%f %f \n',theta)





