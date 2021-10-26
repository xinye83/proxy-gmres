function y = proxy_gmres_apply(A, P, x)
%%proxy_gmres_apply    Apply proxy-GMRES preconditioner
%   y = proxy_gmres_apply(A,P,x) applies the proxy-GMRES polynomial
%   preconditioner for coefficient matrix A to any given vector x.
%
%   Input:
%       A: 2D array or function handle
%           the input can be the matrix A itself of a function handle that
%           computes the matrix-vector multiplication of A to any given
%           vector; the handle must take a single input, the vector x
%           where the matrix is applied to, and return a single vector y
%           that contains the result.
%
%       P: structure
%           the structure returned from proxy_gmres_build which contains
%           all information to performe the matrix-vector multiplication.
%
%       x: 1D array
%           the vector to apply the preconditioner
%
%   Output:
%       y: 1D array
%           the result of the matrix-vector multiplication
%
%   For more details please refer to our paper:
%       Proxy-GMRES: preconditioning via GMRES in polynomial space, Xin Ye,
%       Yuanzhe Xi, and Yousef Saad, SIAM Journal on Matrix Analysis and
%       Applications, 42(3), 1248-1267.

% Author: Xin Ye
% Date: May 2021
% Email: yexinwhu@gmail.com

%%
V = zeros(length(x), length(P.coefficient));
V(:, 1) = x / sqrt(P.numProxyPoints);

for i = 1: length(P.coefficient) - 1
    if ~isa(A, 'function_handle')
        V(:, i + 1) = A * V(:, i);
    else
        V(:, i + 1) = feval(A, V(:, i));
    end
    
    for j = max(1, i - P.recurrenceLength + 1): i
        V(:, i + 1) = V(:, i + 1) - P.upperHessenbergMatrix(j, i) * V(:, j);
    end
    
    V(:, i + 1) = V(:, i + 1) / P.upperHessenbergMatrix(i + 1, i);
end

y = V * P.coefficient;

end
