function W_FDLA=local_degree(P,L,eps_deg)

% References: 
%  W. Shi, Q. Ling, G. Wu, and W. Yin, “EXTRA: An Exact First-Order Algorithm for Decentralized Consensus Optimization,”
%       SIAM Journal on Optimization, vol. 25, no. 2, pp. 944–966, 2015.
%  S. Boyd, P. Diaconis, and L. Xiao, “Fastest mixing markov chain on a graph,” SIAM
% review, vol. 46, no. 4, pp. 667–689, 2004.
W_FDLA(1:L,1:L)=0;
deg=sum(P,2);
for i=1:L
    for j=1:L
        if P(i,j)==1
            W_FDLA(i,j)=1/(max(deg(i),deg(j))+eps_deg);
        end;
    end;
end;
W_FDLA=diag(ones(L,1)-sum(W_FDLA,2))+W_FDLA;
end
