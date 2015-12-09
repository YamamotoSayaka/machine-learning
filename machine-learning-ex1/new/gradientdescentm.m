% data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
cd D:\ml\machine-learning-ex1\new
data = load('ex1data2.txt');
x1 = data(:, 1); x2 = data(:, 2); y = data(:, 3);
m = length(y); 
%feature normalize%%%%%%%%%%%%%%%%%%%%
%mux1 = mean(x1);
%stdx1 = std(x1);
%x1 = (x1 - mux1)/ stdx1;
%mux2 = mean(x2);
%stdx2 = std(x2);
%x2 = (x2 - mux2)/ stdx2;
%X=[ones(m,1) x1 x2];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = data(:, 1:2);
%X_norm = X;

%mu = zeros(1, size(X, 2));
%sigma = zeros(1, size(X, 2));

%mu = mean(X);
%sigma = std(X);

%for i = 1:size(X,2)
%    X_norm(:,i) = (X(:,i) - mu(i)) / sigma(i);
%end
%X = X_norm;
X = [ones(m,1) X];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

theta = [0;0;0];
alpha = 0.01;
h = X * theta;

do
    theta(1) = theta(1) - alpha * (1/m) * sum(h-y);
    theta(2) = theta(2) - alpha * (1/m) * sum((h - y) .* x1);
    theta(3) = theta(3) - alpha * (1/m) * sum((h - y) .* x2);
    theta = [theta(1);theta(2);theta(3)];
    h = X * theta;
    J =  1/(2*m)*(h-y)' * (h-y);  %costfunction
until(J< 5)
fprintf('%f %f %f\n', theta(1), theta(2),theta(3));