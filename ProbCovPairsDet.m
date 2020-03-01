% Simulates a network with transmitter-receiver pairs in order to examine
% the coverage based on the signal-to-interference ratio (SINR).  The
% network has a random medium access control (MAC) scheme based on a
% determinantal point process, as outlined in the paper[1] by
% B\laszczyszyn, Brochard and Keeler. This code validates by simulation
% Lemma II.1 in the paper[1]. This result gives the probability of
% coverage based on the SINR value of a transmitter-receiver pair in
% a non-random network of transmitter-receiver pairs such as a realization
% of a random point process.
%
% The simulation section estimates the empirical probability of SINR-based
% coverage. For a large enough number of simulations, this empirical result
% will agree with the analytic result given in INSERT RESULT in the paper[2].
%
%
% By coverage, it is assumed that the transmitter-receiver pair are active
% *and* the SINR of the transmitter is larger than some threshold at the
% corresponding receiver.
%
% This code was originally written by H.P Keeler for the paper by
%
% If you use this code in published research, please cite paper[1].
%
% References:
%
% [1] B\laszczyszyn, Brochard and Keeler, "Coverage probability in
% wireless networks with determinantal scheduling", 2020.
%
% Author: H. Paul Keeler, 2020.


close all;
clearvars; clc;

%set random seed for reproducibility
%rng(1);

choiceExample=1; %1 or 2 for a random (uniform) or deterministic example

%%%START -- Parameters -- START%%%
numbSim=10^4; %number of simulations

numbPairs=10;%number of pairs

indexTransPair=1; %index for transmitter-receiver pair
% the above index has to be such that 1<=indexTransPair<=numbPairs

%fading model
muFading=1/3; %Rayleigh fading average
%path loss model
betaPath=2; %pathloss exponent
kappaPath=1; %rescaling constant for pathloss function

thresholdSINR=0.1; %SINR threshold value
constNoise=0; %noise constant

%choose kernel
choiceKernel=1; %1 for Gaussian (ie squared exponetial );2 for Cauchy
sigma=1;% parameter for Gaussian and Cauchy kernel
alpha=1;% parameter for Cauchy kernel

%Simulation window parameters
xMin=-1;xMax=1;yMin=-1;yMax=1;
xDelta=xMax-xMin;yDelta=yMax-yMin; %rectangle dimensions
xx0=mean([xMin,xMax]); yy0=mean([yMin,yMax]); %centre of window
%%%END -- Parameters -- END%%%

%Simulate a random point process for the network configuration
%interferer section
if choiceExample==1
    %random (uniform) x/y coordinates
    %transmitters
    xxTX=xDelta*(rand(numbPairs,1))+xMin;
    yyTX=yDelta*(rand(numbPairs,1))+yMin;
    %all receivers
    xxRX=2*xDelta*(rand(numbPairs,1))+xMin;
    yyRX=2*yDelta*(rand(numbPairs,1))+yMin;
else
    %non-random x/y coordinates
    %all transmitters
    t=2*pi*(linspace(0,(numbPairs-1)/numbPairs,numbPairs)');
    xxTX=(1+cos(5*t+1))/2;
    yyTX=(1+sin(3*t+2))/2;
    %all receivers
    xxRX=(1+sin(3*t+1))/2;
    yyRX=(1+cos(7*t+2))/2;
end

%transmitter location
xxTX0=xxTX(indexTransPair);
yyTX0=yyTX(indexTransPair);

%receiver location
xxRX0=xxRX(indexTransPair);
yyRX0=yyRX(indexTransPair);

%%% START -- CREATE L matrix -- START %%%
%all squared distances of x/y difference pairs
xx=xxTX;yy=yyTX;
xxDiff=bsxfun(@minus,xx,xx');
yyDiff=bsxfun(@minus,yy,yy');
rrDiffSquared=(xxDiff.^2+yyDiff.^2);
if choiceKernel==1
    %%Gaussian/squared exponential kernel
    L=exp(-(rrDiffSquared)/sigma^2);
elseif choiceKernel==2
    %%Cauchy kernel
    L=1./(1+rrDiffSquared/sigma^2).^(alpha+1/2);
else
    error('choiceKernel has to be equal to 1 or 2.');
end

L=10*L; %scale matrix up (increases the eigenvalues ie number of points)
%%% END-- CREATE L matrix -- %%% END

%Eigen decomposition of L
[eigenVecL,eigenValL]=eig(L);

%Helper functions
funPathloss=@(r)((kappaPath*(1+r)).^(-betaPath)); %pathloss function
%Functions for the probability of being connected
fun_h=@(s,r)(1./((funPathloss(s)./funPathloss(r))...
    *thresholdSINR+1));
fun_w=@(r)(exp(-(thresholdSINR/muFading)*constNoise./funPathloss(r)));

%%%START Empirical Connection Proability (ie SINR>thresholdConst) START%%%
%initialize  boolean vectors/arrays for collecting statistics
boolePairExists=false(numbSim,1); %transmitter-receiver pair exists
booleCov=false(numbSim,1); %transmitter-receiver pair is connected
booleTX=false(numbSim,1); %transmitter-receiver pair exists
%loop through all simulations
for ss=1:numbSim
    indexDPP=funSimSimpleDPP(eigenVecL,eigenValL);
    %if transmitter-receiver pair exists in the determinantal outcome
    booleTX(ss)=any(indexDPP==indexTransPair);
    
    if booleTX(ss)
        boolePairExists(ss)=true;
        
        %create Boolean variable for active interferers
        booleTransmit=false(numbPairs,1);
        booleTransmit(indexDPP)=true;
        booleTransmit(indexTransPair)=false;
        
        %x/y values of interfering nodes
        xxInter=xxTX(booleTransmit);
        yyInter=yyTX(booleTransmit);
        
        %number of interferers
        numbInter=sum(booleTransmit);
        
        %simulate signal for interferers
        fadeRandInter=exprnd(muFading,numbInter,1); %fading
        distPathInter=hypot(xxInter-xxRX0,yyInter-yyRX0); %path distance
        proplossInter=fadeRandInter.*funPathloss(distPathInter); %pathloss
        
        %simulate signal for transmitter
        fadeRandSig=exprnd(muFading); %fading
        distPathSig=hypot(xxTX0-xxRX0,yyTX0-yyRX0); %path distance
        proplossSig=fadeRandSig.*funPathloss(distPathSig); %pathloss
        
        %Calculate SINR
        SINR=proplossSig/(sum(proplossInter)+constNoise);
        
        %see if transmitter is connected
        booleCov(ss)=SINR>thresholdSINR;
    end
end
%Estimate empirical probabilities
probCovCondEmp=mean(booleCov(booleTX));%SINR>thresholdConst given pair
probTXEmp=mean(booleTX); %transmitter-receiver pair exists
probCovEmp=mean(booleCov); %SINR>thresholdConst

%%%END Empirical Connection Proability (ie SINR>thresholdConst) END%%%

%%%START Numerical Connection Proability (ie SINR>thresholdConst) START%%%
K=funLtoK(L); %caclulate K kernel from kernel L
sizeK=size(K,1); %number of columns/rows in kernel matrix K

%Calculate all respective distances (based on random network configuration)
%transmitters to other receivers
dist_ji_xx=bsxfun(@minus,xxTX,xxRX');
dist_ji_yy=bsxfun(@minus,yyTX,yyRX');
dist_ji=hypot(dist_ji_xx,dist_ji_yy); %Euclidean distances
%transmitters to receivers
dist_ii_xx=xxTX-xxRX;
dist_ii_yy=yyTX-yyRX;
dist_ii=hypot(dist_ii_xx,dist_ii_yy); %Euclidean distances
dist_ii=repmat(dist_ii',sizeK,1);%repeat cols for element-wise evaluation

%apply functions
hMatrix=fun_h(dist_ji,dist_ii); %matrix H for all h_{x_i}(x_j) values
W_x=fun_w(hypot(xxTX-xxRX,yyTX- yyRX)); %noise factor

%create h matrix corresponding to transmitter-receiver pair
booleAll=true(sizeK,1); %Boolean vector for all pairs
booleReduced=booleAll;
booleReduced(indexTransPair)=false;%remove transmitter
booleReduced(indexTransPair)=false;%remove transmitter
%choose transmitter-receiver row
hVectorReduced=hMatrix(booleReduced,indexTransPair);
%repeat vector hVectorReduced as columns
hMatrixReduced=repmat(hVectorReduced,1,sizeK-1);

%create reduced Palm kernels
KPalmReduced=funPalmK(K,indexTransPair); %reduced Palm version of K matrix
%calculate final kernel
KReduced_h=sqrt(1-hMatrixReduced').*KPalmReduced.*sqrt(1-hMatrixReduced);

%calculate unconditional probabiliity for the event that transmitter's
%signal at the receiver has an SINR>tau, given the pair is active (ie
%transmitting and receiving); see equation (??) in [2]
probCovCondEmp
probCovCond=det(eye(sizeK-1)-KReduced_h)*W_x(indexTransPair)

%transmitting probabilities given by diagonals of kernel
probTXEmp
probTX=K(indexTransPair,indexTransPair)

%calculate unconditional probability
probCovEmp
probCov=probTX.*probCovCond
%%%END Numerical Connection Proability END%%%

%%TEST
% [probCov,probTX,probCovCond]=funProbCovPairsDet(xxTX,yyTX,xxRX,yyRX,fun_h,fun_w,L,1)

%%% START -- Plotting -- START %%%
if ~isempty(indexDPP)
    figure;hold on;
        markerSizeNumb=10; %marker size 
        plot(xxTX,yyTX,'ko','MarkerSize',markerSizeNumb); 
        plot(xxRX,yyRX,'kx','MarkerSize',markerSizeNumb);        
        plot([xxTX';xxRX'],[yyTX';yyRX'],'--k');
    for ii=1:length(indexDPP)
        vectorColor=rand(1,3).^(1); %random vector for marker colors 
        %plot transmitter
        plot(xxTX(indexDPP(ii)),yyTX(indexDPP(ii)),'o','color',...
            vectorColor,'MarkerSize',markerSizeNumb);
        %plot receiver
        plot(xxRX(indexDPP(ii)),yyRX(indexDPP(ii)),'x','color',...
            vectorColor,'MarkerSize',markerSizeNumb);
        %plot line connecting transmitter and receiver
        plot([xxTX(indexDPP(ii));xxRX(indexDPP(ii))],....
            [yyTX(indexDPP(ii));yyRX(indexDPP(ii))],'color',vectorColor);
        axis square;set(gca,'YTick',[]); set(gca,'XTick',[]);
    end
    
end
%%% END -- Plotting -- END %%%

