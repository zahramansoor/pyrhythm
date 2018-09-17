function [nquadint] = func_nquadint(N, x1, x2, y1, y2)
% integral of each quad expression of shape functions

syms x y

for i = 1:4
    for j = 1:4
        for k = 1:4
            for l = 1:4
                f_nquadint = N(i) .* N(j) .* N(k) .* N(l);
                nquadint(i,j,k,l) = vpaintegral(int(f_nquadint, x1, x2), y1, y2);
            end
        end
    end
end

end


