function P = proxy_gmres_build(z, m, k)
%%proxy_gmres_build    Construct proxy-GMRES preconditioner
%   P = proxy_gmres_build(z,m,k) builds the proxy-GMRES polynomial
%   preconditioner of degree m-1 with recurrence length k from a set of
%   points z on the complex plane.
%
%   Input:
%       z: 1D array
%           proxy points that approximate the spectral boundary of the
%           coefficient matrix.
%
%       m: interger
%           dimension of the polynomial space where the preconditioner will
%           be constructed, the degree of the polynomial preconditioner is
%           no greater than m-1.
%
%       k: interger
%           recurrence length of the polynomial basis; when k = 0 at input,
%           it is dynamically determined so that the condition number of
%           the polynomial basis is within the preset threshold 10^6.
%
%   Output:
%       P: structure
%           can be used together with proxy_gmres_apply to compute the
%           matrix-vector product of the preconditioner and a vector.
%
%   For more details please refer to our paper:
%       Proxy-GMRES: preconditioning via GMRES in polynomial space, Xin Ye,
%       Yuanzhe Xi, and Yousef Saad, SIAM Journal on Matrix Analysis and
%       Applications, 42(3), 1248-1267.

% Author: Xin Ye
% Date: May 2021
% Email: yexinwhu@gmail.com

%% input checking
if m < 1
    error('Polynomial dimension must be positive.');
end

if k < 0
    error('Recurrence length must be non-negative.');
end

if m >= length(z)
    error(['Polynomial dimension must be smaller than the number of ' ...
        'proxy points.']);
end

%% setup internal variables
P.numProxyPoints = length(z);
proxyPoints = reshape(z, P.numProxyPoints, 1);
polynomialDimension = m;
P.recurrenceLength = k;

dynamicRecurrencLength = false;

if P.recurrenceLength == 0
    dynamicRecurrencLength = true;
    P.recurrenceLength = 1;
    isStable = false;

    % threshold for the condition number of polynomial basis, the default
    % is 10^6 and can be modified by the user if needed
    condionNumberThreshold = 1e6;
elseif P.recurrenceLength >= polynomialDimension
    % recurrence length too large, use full orthogonalization
    P.recurrenceLength = polynomialDimension;
end

%% Arnoldi-like process
if ~dynamicRecurrencLength
    [P.upperHessenbergMatrix, polynomialBasis, ~] = ...
        polynomial_arnoldi(proxyPoints, polynomialDimension, ...
        P.recurrenceLength, 0);
else
    while ~isStable
        P.recurrenceLength = P.recurrenceLength + 1;
        [P.upperHessenbergMatrix, polynomialBasis, isStable] = ...
            polynomial_arnoldi(proxyPoints, polynomialDimension, ...
            P.recurrenceLength, condionNumberThreshold);
    end
end

%% solve for coefficients
if P.recurrenceLength == polynomialDimension
    % orthogonal basis
    v = zeros(polynomialDimension + 1, 1); v(1) = sqrt(P.numProxyPoints);
    P.coefficient = P.upperHessenbergMatrix \ v;
else
    % short recurrence, non-orthogonal basis
    v = ones(P.numProxyPoints, 1);
    P.coefficient = (polynomialBasis * P.upperHessenbergMatrix) \ v;
end

P.upperHessenbergMatrix = P.upperHessenbergMatrix(1: m, 1: m - 1);

end
