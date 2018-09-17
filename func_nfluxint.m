function [nfluxint] = func_nfluxint(N, x1, x2, y1, y2)
% integral of each flux expression of the shape functions

syms x y

for i = 1:4
    for j = 1:4
        f_nfluxint_x = N(i).*gradient(N(j), x);
        f_nfluxint_y = N(i).*gradient(N(j), y);
        nfluxint(i,j,1) = -int(subs(f_nfluxint_y, y, y1), x, x1, x2); % Edge 1 - bottom boundary
        nfluxint(i,j,2) = int(subs(f_nfluxint_x, x, x2), y, y1, y2); % Edge 2 - right boundary
        nfluxint(i,j,3) = int(subs(f_nfluxint_y, y, y2), x, x2, x1); % Edge 3 - top boundary
        nfluxint(i,j,4) = -int(subs(f_nfluxint_x, x, x1), y, y2, y1); % Edge 4 - left boundary
    end
end

end


