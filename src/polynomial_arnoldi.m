function [H, Q, s] = polynomial_arnoldi(z, m, k, tol)
% Helper function to proxy_gmres_build:
%   Run an Arnoldi-like process in the polynomial space and return the
%   upper-Hessenberg matrix and the polynomial basis, also check if the
%   basis has condition number within the given shreshold if recurrence
%   length is dynamically determined.

% Author: Xin Ye
% Date: May 2021
% Email: yexinwhu@gmail.com

n = length(z);
checkConditionNumber = (tol ~= 0);

H = zeros(m + 1, m);
Q = zeros(n, m + 1);
s = true;

if checkConditionNumber
    G = eye(m);
    L = zeros(m);
    L(1, 1) = 1;
end

Q(:, 1) = ones(n, 1) / sqrt(n);

for j = 1: m
    q = z .* Q(:, j);
    
    for i = max(1, j - k + 1): j
        H(i, j) = Q(:, i)' * q;
        q = q - H(i, j) * Q(:, i);
    end
    
    H(j + 1, j) = norm(q);
    Q(:, j + 1) = q / H(j + 1, j);
    
    if j < m && checkConditionNumber
        % compute new entries in Gram matrix (upper triangular part)
        for i = 1: j
            G(i, j + 1) = Q(:, j + 1)' * Q(:, i);
        end
        
        % get condition number from the Cholesky factor
        L(1: j, j + 1) = L(1: j, 1: j) \ Q(1: j, j + 1);
        L(j + 1, j + 1) = sqrt(1 - L(1: j, j + 1)' * L(1: j, j + 1));
        conditionNumber = cond(L(1: j + 1, 1: j + 1));
        
        if conditionNumber >= tol
            s = false;
            return;
        end
    end
end
