
%
%This code is built based on the code written by Mihai Alin Baidu, and is written by QI Zhang and Jiang Zhu. If you have any problems, please feel free to contact jiangzhu16@zju.edu.cn

clear variables
% rng('default');
rng(1);

%% Generate a noisy superposition of K complex sinusoids with angular frequencies in [-pi,pi)
N    = 20;          % size of the full data
M    = 20;          % number of measurements with indices randomly chosen from 0,...,N-1
K    = 5;           % number of sinusoidal components
G    = 10;          % number of snapshots
d    = 2*pi/N;      % minimum separation of the angular frequencies
SNR  = 0;          % signal to noise ratio (dB)

% Generate random measurement indices
tmp  = randperm(N-1)';
Mcal = [sort(tmp(1:M-1),'ascend'); N]-1;    % indices of the measurements

% Generate K frequencies with minimum separation d
omega = zeros(K,1);
omega(1) = pi * (2*rand - 1);
for k = 2:K
    th = pi * (2*rand - 1);
    while min(abs((wrapToPi(th-omega(1:k-1))))) < d
        th = pi * (2*rand - 1);
    end
    omega(k) = th;
end

A   = exp(1i*(0:1:Mcal(end)).'*omega.');         % matrix with columns a(omega(1)),..., a(omega(K))
R   = 1 + .2.*randn(K,G);                        % magnitudes of the complex amplitudes
W  = R.*exp(1i*2*pi*rand(K,G));                 % complex amplitudes   
X   = A*W;                                      % original signal
Pn  = mean(mean(abs(X(Mcal+1,:)).^2))*10^(-SNR/10);       % noise power
eps = sqrt(0.5*Pn).*(randn(M,G)+1i*randn(M,G));  % complex Gaussian noise
Y   = X(Mcal+1,:) + eps;                           % measured signal

% plot the line spectral signal
figure; hold on;
axis([-pi pi -inf inf]); xlabel('Frequencies'); ylabel('Magnitudes');
stem(omega,mean(abs(W),2),'*k'); 

%% MVALSE (with Heuristic 2)
tic;
outMultiVALSEBadiu = MVALSE( Y, Mcal, 2, X );
toc
fprintf('Runtime of MultiVALSE: %g s.\n',toc);
stem(outMultiVALSEBadiu.freqs,mean(abs(outMultiVALSEBadiu.amps),2),'or'); 

