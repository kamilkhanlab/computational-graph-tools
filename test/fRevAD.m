% Computes y = f(x) and xBar = (Df(x))'*yBar, using the reverse mode of
% automatic differentiation (AD). 
%
% x, y, xBar, and yBar are all column vectors of appropriate dimension.
% If yBar = [1.0], then xBar will be the gradient vector of f at x.
%
% This code was automatically generated by ReverseAD.jl.
function [y, xBar] = fRevAD(x, yBar)
% initialize
l = 12;  % tape length
v = zeros(l, 1);  % values at each node of computational graph
vBar = zeros(size(v));  % adjoints at each node of computational graph
y = zeros(size(yBar));  % will be f(x)
xBar = zeros(size(x));  % will be (Df(x))'*yBar

% evaluate y with forward sweep through computational graph
v(1) = x(1);
v(2) = x(2);
v(3) = 1.0;
v(4) = v(3) - v(1);
v(5) = v(4).^2;
v(6) = v(1).^2;
v(7) = v(2) - v(6);
v(8) = v(7).^2;
v(9) = 100.0;
v(10) = v(9) * v(8);
v(11) = v(5) + v(10);
v(12) = v(11);
y(1) = v(12);

% evaluate xBar with reverse sweep through computational graph
vBar(12) = yBar(1);
vBar(11) = vBar(11) + vBar(12);
vBar(5) = vBar(5) + vBar(11);
vBar(10) = vBar(10) + vBar(11);
vBar(9) = vBar(9) + vBar(10) * v(8);
vBar(8) = vBar(8) + vBar(10) * v(9);
vBar(7) = vBar(7) + vBar(8) * 2 * v(7).^(1);
vBar(2) = vBar(2) + vBar(7);
vBar(6) = vBar(6) - vBar(7);
vBar(1) = vBar(1) + vBar(6) * 2 * v(1).^(1);
vBar(4) = vBar(4) + vBar(5) * 2 * v(4).^(1);
vBar(3) = vBar(3) + vBar(4);
vBar(1) = vBar(1) - vBar(4);
xBar(2) = vBar(2);
xBar(1) = vBar(1);

return
