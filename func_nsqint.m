function [nsqint] = func_nsqint(N, x1, x2, y1, y2)
% integral of each quadratic expression of the shape functions

syms x y

for i = 1:4
    for j = 1:4
        f_nsqint = N(i) .* N(j);
        nsqint(i,j) = vpaintegral(int(f_nsqint, x1, x2), y1, y2);
    end
end

end


