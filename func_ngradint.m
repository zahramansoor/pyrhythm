function [ngradint] = func_ngradint(N, x1, x2, y1, y2)
% integral of each gradient expression of the shape functions

syms x y

for i = 1:4
    for j = 1:4
        f_ngradint = dot(gradient(N(i), [x,y]), 0.1*gradient(N(j), [x,y]));
        ngradint(i,j) = vpaintegral(int(f_ngradint, x, x1, x2), y1, y2);
    end
end

end


