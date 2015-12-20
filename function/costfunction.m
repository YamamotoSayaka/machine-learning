function [j,g] = costfunction(theta)
j = (theta(1)-5)^2+(theta(2)-5)^2;
g = zeros(2,1);
g(1) = 2*(theta(1)-5);
g(2) = 2*(theta(2)-5);