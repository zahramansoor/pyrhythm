function [ncubint] = func_ncubint(N, x1, x2, y1, y2)
% integral of each cubic expression of shape functions

syms x y

for i = 1:4
    for j = 1:4
        for k = 1:4
            f_ncubint = N(i) .* N(j) .* N(k);
            ncubint(i,j,k) = vpaintegral(int(f_ncubint, x1, x2), y1, y2);
        end
    end
end

end


