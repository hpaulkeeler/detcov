% This code thins a point process using a determinantal thinning. Then the
% various Palm results are checked empirically against numerical results. 
%
% This code was originally written by H.P Keeler for the paper by
% B\laszczyszyn, Brochard and Keeler[1].
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
rng(1);

choiceExample=1; %1 or 2 for a random (uniform) or deterministic example

%%%START -- Parameters -- START%%%
numbSim=10^4; %number of simulations

numbPoints=10;%number of pairs

indexX=1; %index for point X
indexY=2; %index for point Y
indexConfig=[3,4]; %an index for a test configuration
%above indices are bounded from above by numbNodes

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
    %transmitters or receivers
    xx=xDelta*(rand(numbPoints,1))+xMin;
    yy=yDelta*(rand(numbPoints,1))+yMin;
else
    %non-random x/y coordinates
    %all transmitters
    t=2*pi*(linspace(0,(numbPoints-1)/numbPoints,numbPoints)');
    xx=(1+cos(5*t+1))/2;
    yy=(1+sin(3*t+2))/2;
end

%%% START -- CREATE L matrix -- START %%%
%all squared distances of x/y difference pairs
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
%%% END-- CREATE L matrix -- %%% END

%Eigen decomposition of L
[eigenVecL,eigenValL]=eig(L);

%%%START Empirical Connection Proability (ie SINR>thresholdConst) START%%%
%initialize  boolean vectors/arrays for collecting statistics
booleExistX=false(numbSim,1); %first point exist
booleExistY=false(numbSim,1); %second point exist
boolePairExists=false(numbSim,1); %transmitter-receiver pair exists
booleConfig=false(numbSim,1); %configuration exists
numbConfig=length(indexConfig); %number of points in configuration
%loop through all simulations
for ss=1:numbSim
    indexDPP=funSimSimpleDPP(eigenVecL,eigenValL);
    
    booleExistX(ss)=any(indexDPP==indexX);
    booleExistY(ss)=any(indexDPP==indexY);
    
    indexTemp=setdiff(indexDPP,[indexX,indexY]); %remove X and Y
    
    %see if configuration exists
    booleConfig(ss)=sum(ismember(indexConfig,indexTemp))==numbConfig;
    
end
booleExistXY=booleExistX&booleExistY;
%%%END Empirical Connection Proability (ie SINR>thresholdConst) END%%%

%%%START Numerical Connection Proability (ie SINR>thresholdConst) START%%%
K=funLtoK(L); %caclulate K kernel from kernel L
sizeK=size(K,1); %number of columns/rows in kernel matrix K

%Probabity of single points
probXEmp=mean(booleExistX)
probX=K(indexX,indexX)

probYEmp=mean(booleExistY)
probY=K(indexY,indexY)

probConfigEmp=mean(booleConfig)
probConfig=det(K(indexConfig,indexConfig))

%probability of configuration conditioned on a single point
probConfigXEmp=mean(booleConfig(booleExistX))
[KPalmReducedX,KPalmX]=funPalmK(K,indexX);
probConfigX=det(KPalmX(indexConfig,indexConfig))

probConfigYEmp=mean(booleConfig(booleExistY))
[KPalmReducedY,KPalmY]=funPalmK(K,indexY);
probConfigY=det(KPalmY(indexConfig,indexConfig))

%probability of configuration conditioned on two points
probConfigXYEmp=mean(booleConfig(booleExistXY))
[KPalmReducedXY,KPalmXY]=funPalmK(KPalmX,indexY);
probConfigXY=det(KPalmXY(indexConfig,indexConfig))

[KPalmReducedYX,KPalmYX]=funPalmK(KPalmY,indexX);
probConfigYX=det(KPalmYX(indexConfig,indexConfig))

%probability of point Y existing given X
probYGivenXEmp=mean(booleExistY(booleExistX))
probYGivenX=det(KPalmX(indexY,indexY))

%probability of point X existing given Y
probXGivenYEmp=mean(booleExistY(booleExistX))
probXGivenY=det(KPalmY(indexX,indexX))

%probability of configuration conditioned on point X existing and point Y
%not existing
probConfigXNotYEmp=mean(booleConfig(booleExistX&(~booleExistY)))
probConfigXNotY=(probConfigX-probYGivenX*probConfigXY)/(1-probYGivenX)

%%%END Numerical Connection Proability END%%%

if ~isempty(indexDPP)
    %%% START -- Plotting -- START %%%
    figure;hold on;
    markerSizeNumb=80; %marker size of markers colors
    vectorColor=rand(1,3).^(1); %random vector for colors of
    %Plot point process
    plot(xx,yy,'ko','MarkerSize',markerSizeNumb/6);
    %Plot determinantally-thinned point process
    plot(xx(indexDPP),yy(indexDPP),'.','MarkerSize',markerSizeNumb/3,'color',vectorColor);
    grid;
    axis square;set(gca,'YTick',[]); set(gca,'XTick',[]);
    legend('Original point process', 'Determinantal subset');
    %%% END -- Plotting -- END %%%
end


