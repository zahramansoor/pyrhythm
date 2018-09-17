function [nint] = func_nint(N, x1, x2, y1, y2)
% integral of each shape functions

syms x y

intN1 = vpaintegral(int(N(1), x, x1, x2), y1, y2);
intN2 = vpaintegral(int(N(2), x, x1, x2), y1, y2);
intN3 = vpaintegral(int(N(3), x, x1, x2), y1, y2);
intN4 = vpaintegral(int(N(4), x, x1, x2), y1, y2);

nint = [intN1, intN2, intN3, intN4];

end

