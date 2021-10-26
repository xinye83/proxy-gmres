% main driver for the example of using proxy-GMRES polynomial
% preconditioner on a toy problem, for more details please refer to section
% 5.1 of our paper:
%   Proxy-GMRES: preconditioning via GMRES in polynomial space, Xin Ye,
%   Yuanzhe Xi, and Yousef Saad, SIAM Journal on Matrix Analysis and
%   Applications, 42(3), 1248-1267.

% Author: Xin Ye
% Date: May 2021
% Email: yexinwhu@gmail.com

close all
addpath ../src

%% load the following data into workspace
%   n: dimension of the linear system
%   e: eigenvalues of matrix A
%   z: proxy points on the spectral boundary of matrix A
load data.mat

%% problem setup
% Note that the exact solution to the linear system Ax = b is x = ones(n, 1)
A = diag(e);
b = e;

opts.tolIts = 1e-12;
opts.maxits = 1000;
opts.im = 50;
opts.outputG = 0;

%% GMRES without preconditioner
x1 = zeros(n, 1);

tstart = tic;
[x1, res1, its1] = fgmrez(n, A, @(x) x, b, x1, opts);
t1 = toc(tstart);

%% GMRES with proxy-GMRES polynomial preconditioner
x2 = zeros(n, 1);

tstart = tic;
P = proxy_gmres_build(z, 30, 2);
t2 = toc(tstart);

tstart = tic;
[x2, res2, its2] = fgmrez(n, A, @(x) proxy_gmres_apply(A, P, x), b, x2, opts);
t3 = toc(tstart);

%% plot eigenvalues and proxy points
figure(1);
plot(real(e), imag(e), '.r', real(z), imag(z), '.b', ...
    'MarkerSize',10);
legend('eigenvalues', 'proxy points');
set(gca,'fontsize',14);
axis equal;
xlim([-2.5, 2.5]);
ylim([-0.5, 2.5]);

%% plot the contour map of the filtering by the preconditioner
x = (-2.5: 0.01: 2.5);
y = (-0.5: 0.01: 2.5);
[X, Y] = meshgrid(x, y);
Z = X + 1i * Y;
N = numel(Z);
f = reshape(Z, N, 1);
M = sparse((1: N), (1: N), f, N, N, N);
f = log10(abs(1 - proxy_gmres_apply(M, P, f)));
f = reshape(f, length(y), length(x));

figure(2);
[C,H] = contour(X, Y, f,[(0: -1: -3), 0], ...
    'ShowText', 'on', 'LineWidth', 2);
clabel(C,H,'FontSize',14);
colormap([1 0 1; 0 0 1; 1 0 0; 0 0 0]);
hold on;
plot(0, 0, 'r.', 'MarkerSize', 30);
hold off;
axis equal;
set(gca, 'fontsize', 14);
