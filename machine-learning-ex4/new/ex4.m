tic
figure;load('ex4data1.mat');load('ex4weights.mat');
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10  
m = 5000;lambda = 10;i = 1;n =1;alpha = 2.3333
Theta1g = zeros(size(Theta1));
Theta2g = zeros(size(Theta2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%
yVec = zeros(m,num_labels);
for i = 1:m
    yVec(i,y(i)) = 1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%% fp
a1 = [ones(m,1),X];z2 = a1 *Theta1';a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1),a2];z3 = a2 *Theta2';a3 = sigmoid(z3);
J = 1/m * sum(sum(-1 * yVec .* log(a3)-(1-yVec) .* log(1-a3))) + (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));
%%%%%%%%%%%%%%%%%%%%%%%%%%% bp
do
    for t = 1 : m      %对每个元素进行BP
      a1 = [1; X(t,:)']; %取出X中的第一行 转置后添加偏置项
      z2 = Theta1 * a1;
      a2 = [1;sigmoid(z2)];
      z3 = Theta2 * a2;
      a3 = [1;sigmoid(z3)];
      yn = ([1:10]==y(t))'; %将1-10的y值矩阵表示
      a3 = a3(2:end);
      %%%%%%%%%%%%%%%BP
      d3 = a3 - yn;
      d2 = Theta2'*d3 .*[1;sigmoidG(z2)];
      d2 = d2(2:end); %取出偏置项 
      Theta1g = Theta1g +d2*a1';
      Theta2g = Theta2g +d3*a2';
    end
    Theta1g = (1/m) * Theta1g + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
    Theta2g = (1/m) * Theta2g + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
    Theta1 = Theta1 -alpha*Theta1g;
    Theta2 = Theta2 -alpha*Theta2g;
    n = n + 1;
    a1 = [ones(m,1),X];z2 = a1 *Theta1';a2 = sigmoid(z2);
    a2 = [ones(size(a2,1),1),a2];z3 = a2 *Theta2';a3 = sigmoid(z3); 
    J = 1/m * sum(sum(-1 * yVec .* log(a3)-(1-yVec) .* log(1-a3))) + (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));
    hold on;
    plot(n,J,'bk');
 until n == 4000  
a1 = [ones(m,1),X];z2 = a1 *Theta1';a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1),a2];z3 = a2 *Theta2';a3 = sigmoid(z3); 
J = 1/m * sum(sum(-1 * yVec .* log(a3)-(1-yVec) .* log(1-a3))) + (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m)); 
toc