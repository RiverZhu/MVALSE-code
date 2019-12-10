function out = MVALSE( Y, m, ha, X)
%MVALSE algorithm for line spectral estimation
% INPUTS:
%   Y  - measurement vector of size M
%   m  - is a vector containing the indices (in ascending order) of the M
%       measurements; subset of {0,1,...,m(end)}
%   ha - indicator determining which approximation of the
%       frequency posterior pdfs will be used:
%           ha=1 will use Heuristic #1
%           ha=2 will use Heuristic #2
%           ha=3 will use point estimation of the frequencies (VALSE-pt)
%   x  - the true signal - used for computing the MSE vs iterations
%   This code is built based on the code written by Mihai Alin Baidu, and is written by QI Zhang and Jiang Zhu.

[M,G] = size(Y);
N     = m(M)+1;     % size of full data
Y2    = Y'*Y;
L     = N;          % assumed number of components
A     = zeros(L,L);
J     = zeros(L,L);
H     = zeros(L,G);
W     = zeros(L,G);
C     = zeros(L);
T     = 5000;       % max number of iterations (5000 is very conservative, typically converges in tens of iterations)
mse   = zeros(T,1);
Kt    = zeros(T,1);
t     = 1;

% Initialization of the posterior pdfs of the frequencies
res   = Y;
for l=1:L
    % noncoherent estimation of the pdf
    YI = zeros(N,G);
    YI(m+1,:) = res;
    R  = YI*YI';
    sR = zeros(N-1,1);
    for i=2:N
        for k=1:i-1
            sR(i-k) = sR(i-k) + R(i,k);
        end
    end
    if l==1 % use the sample autocorrelation to initialize the model parameters
        Rh  = toeplitz([sum(diag(R));sR])/N;        
        evs = sort(real(eig(Rh)),'ascend');
        nu  = mean(evs(1:floor(N/4)))/G;
        K   = floor(L/2);
        rho = K/L;
        tau = (trace(Y2)/M-nu*G)/(rho*L);
    end
    etaI   = 2*sR/(M+nu/tau)/nu;
    ind    = find(abs(etaI)>0);
    if ha~=3
        [~,mu,kappa] = Heuristic2(etaI(ind), ind);
        A(m+1,l) = exp(1i*m * mu) .* ( besseli(m,kappa,1)/besseli(0,kappa,1) );
    else
        [~,mu] = pntFreqEst(etaI(ind), ind);
        A(m+1,l) = exp(1i*m * mu);
    end
    
    % compute weight estimates; rank one update
    W_temp = W(1:l-1,:); C_temp = C(1:l-1,1:l-1);
    J(1:l-1,l) = A(m+1,1:l-1)'*A(m+1,l); J(l,1:l-1) = J(1:l-1,l)'; J(l,l) = M;
    H(l,:) = A(m+1,l)'*Y;
    v = nu / ( M + nu/tau - real(J(1:l-1,l)'*C_temp*J(1:l-1,l))/nu );
    u = v .* (H(l,:) - J(1:l-1,l)'*W_temp)/nu;
    W(l,:) = u;
    ctemp = C_temp*J(1:l-1,l)/nu;
    W(1:l-1,:) = W_temp - ctemp*u;
    C(1:l-1,1:l-1) = C_temp + v*(ctemp*ctemp');
    C(1:l-1,l) = -v*ctemp;  C(l,1:l-1) = C(1:l-1,l)'; C(l,l) = v;
    
    % the residual signal
    res = Y - A(m+1,1:l)*W(1:l,:);
    
    if l==K % save mse and K at initialization
        xro    = A(:,1:l)*W(1:l,:);
        mse(t) = norm(X - xro)^2/norm(X)^2;
        Kt(t)  = K;
    end
end

%%% Start the VALSE algorithm
cont = 1;
while cont
% for iter_outter = 1:max_outer
    t = t + 1;
    % Update the support and weights
    [ K, s, W, C ] = maxZ( J, H, M, nu, rho, tau );
    % Update the noise variance, the variance of prior and the Bernoulli probability
    if K>0
        nu  = real(trace(Y2 - 2*real(H(s,:)'*W(s,:)))/G + trace(W(s,:)'*J(s,s)*W(s,:))/G + trace(J(s,s)*C(s,s)) )/M;
        tau = real(trace(W(s,:)*W(s,:)') + G*trace(C(s,s)))/(K*G);
        if K<L
            rho = K/L;
        else
            rho = (L-1)/L; % just to avoid the potential issue of log(1-rho) when rho=1
        end
    else
        rho = 1/L; % just to avoid the potential issue of log(rho) when rho=0
    end
    
    % Update the posterior pdfs of the frequencies
    inz = 1:L; inz = inz(s); % indices of the non-zero components
    th = zeros(K,1);
    for i = 1:K
        if K == 1
            R = Y;
            eta = 2/nu * ( R * W(inz,:)' );
        else
            A_i = A(m+1,inz([1:i-1 i+1:end]));
            R = Y - A_i*W(inz([1:i-1 i+1:end]),:);
            eta = 2/nu * ( R * W(inz(i),:)' - A_i * C(inz([1:i-1 i+1:end]),i) );
        end
        if ha == 1
            [A(:,inz(i)), th(i)] = Heuristic1( eta, m, 1000 );
        elseif ha == 2
            [A(:,inz(i)), th(i)] = Heuristic2( eta, m );
        elseif ha == 3
            [A(:,inz(i)), th(i)] = pntFreqEst( eta, m );
        end
    end
    J(:,s) = A(m+1,:)'*A(m+1,s);
    J(s,:) = J(:,s)';
    J(s,s) = J(s,s) - diag(diag(J(s,s))) + M*eye(K);
    H(s,:)   = A(m+1,s)'*Y;
    
    % stopping criterion:
    % the relative change of the reconstructed signalis below threshold or
    % max number of iterations is reached
    xr     = A(:,s)*W(s,:);
    mse(t) = norm(xr-X)^2/norm(X)^2;
    Kt(t)  = K;
    if (norm(xr-xro)/norm(xro)<1e-6) || (norm(xro)==0&&norm(xr-xro)==0) || (t >= T)
        cont = 0;
        mse(t+1:end) = mse(t);
        Kt(t+1:end)  = Kt(t);
    end
    xro = xr;
end
out = struct('freqs',th,'amps',W(s,:),'x_estimate',xr,'noise_var',nu,'iterations',t,'mse',mse,'K',Kt);

end

function [a, theta, kappa, mu] = Heuristic1( eta, m, D )
%Heuristic1 Uses the mixture of von Mises approximation of frequency pdfs
%and Heuristic #1 to output a mixture of max D von Mises pdfs

M     = length(m);
tmp   = abs(eta);
A     = besseli(1,tmp,1)./besseli(0,tmp,1);
kmix  = Ainv( A.^(1./m.^2) );
[~,l] = sort(kmix,'descend');
eta_q = 0;
for k=1:M
    if m(l(k)) ~= 0
        if m(l(k)) > 1
            mu2   = ( angle(eta(l(k))) + 2*pi*(1:m(l(k))).' )/m(l(k));
            eta_f = kmix(l(k)) * exp( 1i*mu2 );
        else
            eta_f = eta(l(k));
        end
        eta_q = bsxfun(@plus,eta_q,eta_f.');
        eta_q = eta_q(:);
        kappa = abs(eta_q);
        
        % to speed up, use the following 4 lines to throw away components
        % that are very small compared to the dominant one
        kmax  = max(kappa);
        ind   = (kappa > (kmax - 30) ); % corresponds to keeping those components with amplitudes divided by the highest amplitude is larger than exp(-30) ~ 1e-13
        eta_q = eta_q(ind);
        kappa = kappa(ind);
        
        if length(eta_q) > D
            [~, in] = sort(kappa,'descend');
            eta_q   = eta_q(in(1:D));
        end
    end
end
kappa   = abs(eta_q);
mu      = angle(eta_q);
kmax    = max(kappa);
I0reg   = besseli(0,kappa,1) .* exp(kappa-kmax);
Zreg    = sum(I0reg);
n       = 0:1:m(end);
[n1,k1] = meshgrid(n, kappa);
a       = sum( (diag(exp(kappa-kmax))* besseli(n1,k1,1) /Zreg ).*exp(1i*mu*n),1).';
theta   = angle(sum( (diag(exp(kappa-kmax))* besseli(1,kappa,1) /Zreg ).*exp(1i*mu*1),1));
end

function [a, theta, kappa] = Heuristic2( eta, m )
%Heuristic2 Uses the mixture of von Mises approximation of frequency pdfs
%and Heuristic #2 to output one von Mises pdf

N     = length(m);
ka    = abs(eta);
A     = besseli(1,ka,1)./besseli(0,ka,1);
kmix  = Ainv( A.^(1./m.^2) );
k     = N;
eta_q = kmix(k) * exp( 1i * ( angle(eta(k)) + 2*pi*(1:m(k)).' )/m(k) );
for k = N-1:-1:1
    if m(k) ~= 0
        phi   = angle(eta(k));
        eta_q = eta_q + kmix(k) * exp( 1i*( phi + 2*pi*round( (m(k)*angle(eta_q) - phi)/2/pi ) )/m(k) );
    end
end
[~,in] = max(abs(eta_q));
mu     = angle(eta_q(in));
d1     = -imag( eta' * ( m    .* exp(1i*m*mu) ) );
d2     = -real( eta' * ( m.^2 .* exp(1i*m*mu) ) );
if d2<0 % if the function is locally concave (usually the case)
    theta  = mu - d1/d2;
    kappa  = Ainv( exp(0.5/d2) );
else    % if the function is not locally concave (not sure if ever the case)
    theta  = mu;
    kappa  = abs(eta_q(in));
end
n      = (0:1:m(end))';
a      = exp(1i*n * theta).*( besseli(n,kappa,1)/besseli(0,kappa,1) );
end

function [a, theta] = pntFreqEst( eta, m )
%pntFreqEst - point estimation of the frequency

th     = -pi:2*pi/(100*max(m)):pi;

[~,i]  = max(real( eta'*exp(1i*m*th) ));
mu     = th(i);
d1     = -imag( eta' * ( m    .* exp(1i*m*mu) ) );
d2     = -real( eta' * ( m.^2 .* exp(1i*m*mu) ) );
if d2<0 % if the function is locally concave (usually the case)
    theta  = mu - d1/d2;
else    % if the function is not locally concave (not sure if ever the case)
    theta  = mu;
end
a      = exp(1i*(0:1:m(end))' * theta);
end

function [ K, s, W, C ] = maxZ( J, H, M, nu, rho, tau )
%maxZ maximizes the function Z of the binary vector s, see Appendix A of
%the paper

[L,G] = size(H);
cnst = log(rho/(1-rho));

K = 0; % number of components
s = false(L,1); % Initialize s
W = zeros(L,G);
C = zeros(L);
U = zeros(L,G);
v = zeros(L,1);
Delta = zeros(L,1);
if L > 1
    cont = 1;
    while cont
        if K<M-1
            v(~s) = nu ./ ( M + nu/tau - real(sum(J(s,~s).*conj(C(s,s)*J(s,~s)),1))/nu );
            U(~s,:) = repmat(v(~s),1,G) .* ( H(~s,:) - J(s,~s)'*W(s,:))/nu;
            Delta(~s) = G*log(v(~s)./tau) + diag(U(~s,:)*U(~s,:)')./v(~s) + cnst;
        else
            Delta(~s) = -1; % dummy negative assignment to avoid any activation
        end
        if ~isempty(H(s,:))
            Delta(s) = -G*log(diag(C(s,s))./tau) - diag(W(s,:)*W(s,:)')./diag(C(s,s)) - cnst;
        end
        [~, k] = max(real(Delta));
        if Delta(k)>0
            if s(k)==0 % activate
                W(k,:) = U(k,:);
                ctemp = C(s,s)*J(s,k)/nu;
                W(s,:) = W(s,:) - ctemp*U(k,:);
                C(s,s) = C(s,s) + v(k)*(ctemp*ctemp');
                C(s,k) = -v(k)*ctemp;
                C(k,s) = C(s,k)';
                C(k,k) = v(k);
                s(k) = ~s(k); K = K+1;
            else % deactivate
                s(k) = ~s(k); K = K-1;
                W(s,:) = W(s,:) - C(s,k)*W(k,:)/C(k,k);
                C(s,s) = C(s,s) - C(s,k)*C(k,s)/C(k,k);
            end
            C = (C+C')/2; % ensure the diagonal is real
        else
            break
        end
    end
elseif L == 1
    if s == 0
        v = nu ./ ( M + nu/tau );
        U = v * H/nu;
        Delta = log(v) + u*conj(u)/v + cnst;
        if Delta>0
            W = U; C = v; s = 1; K = 1;
        end
    else
        Delta = -log(C) - W*W'/C - cnst;
        if Delta>0
            W = 0; C = 0; s = 0; K = 0;
        end
    end
end
end

function [ k ] = Ainv( R )
% Returns the approximate solution of the equation R = A(k),
% where A(k) = I_1(k)/I_0(k) is the ration of modified Bessel functions of
% the first kind of first and zero order
% Uses the approximation from
%       Mardia & Jupp - Directional Statistics, Wiley 2000, pp. 85-86.
%
% When input R is a vector, the output is a vector containing the
% corresponding entries

k   = R; % define A with same dimensions
in1 = (R<.53); % indices of the entries < .53
in3 = (R>=.85);% indices of the entries >= .85
in2 = logical(1-in1-in3); % indices of the entries >=.53 and <.85
R1  = R(in1); % entries < .53
R2  = R(in2); % entries >=.53 and <.85
R3  = R(in3); % entries >= .85

% compute for the entries which are < .53
if ~isempty(R1)
    t      = R1.*R1;
    k(in1) = R1 .* ( 2 + t + 5/6*t.*t );
end
% compute for the entries which are >=.53 and <.85
if ~isempty(R2)
    k(in2) = -.4 + 1.39*R2 + 0.43./(1-R2);
end
% compute for the entries which are >= .85
if ~isempty(R3)
    k(in3) = 1./( R3.*(R3-1).*(R3-3) );
end

end