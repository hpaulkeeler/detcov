
% [1] Lavancier, Moller and Rubak, "Determinantal point process models and
% statistical inference", 2015
% [2] Biscio and Lavancier, "Quantifying repulsiveness of determinantal
% point processes", 2016


function S=funS(xx,yy,choiceKernel,sigma)
%xx/yy need to be column vectors
xx=xx(:);
yy=yy(:);

sizeS=length(xx); %number of columns/rows

%%%NOTE:
% As sigma approaches zero, S approches the identity matrix
% As sigma approaches infinity, S approches a matrix of ones, which has a
% zero determinant (meaning its ill-conditioned in terms of inverses)

%%%START - Create similarity matrix S - START%%%
if sigma~=0
    %all squared distances of x/y difference pairs
    xxDiff=bsxfun(@minus,xx,xx'); yyDiff=bsxfun(@minus,yy,yy');
    rrDiffSquared=(xxDiff.^2+yyDiff.^2);
    if choiceKernel==1
        %%Gaussian kernel
        %See the paper by Lavancier, Moller and Rubak (2015)
        S=exp(-(rrDiffSquared)/sigma^2);
    elseif choiceKernel==2
        %%Cauchy kernel
        %See the paper by Lavancier, Moller and Rubak (2015)
        alpha=1; %an additional parameter for the Cauchy (ie second) kernel
        S=1./(1+rrDiffSquared/sigma^2).^(alpha+1/2);
    elseif choiceKernel==3
        %%Bessel kernel
        %See the Supplementary Material for the paper by  Biscio and
        %Lavancier (2016), page 2007. Kernel CI, where sigma has been
        %introduced as a scale parameter, similar to the Gaussian and
        %Cauchy cases.
        rrDiff=sqrt(rrDiffSquared);
        rrDiff(1:1+size(rrDiff,1):end)=1; %prevent zero division
        %Bessel (simplified) kernel
        S=besselj(1,2*sqrt(pi)*rrDiff/sigma)./(sqrt(pi)*rrDiff/sigma);
        %need to rescale to ensure that diagonal entries are ones.
        S(1:1+size(S,1):end)=1; %set to correct value
    end
else
    S=eye(sizeS);
end
%%%END - Create similarity matrix S - END%%%
end
