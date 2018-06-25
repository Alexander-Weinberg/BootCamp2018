# Quesiton 2
%% PARAMETERS

% preferences
gamma   = 2
beta    = 0.95;

%returns
r           = 0.03;
R = 1+ r;

% income risk: discretized N(mu,sigma^2)
mu_y    = 1;
sd_y    = 0.2;
ny      = 5;

% asset grids
na          = 30;
amax        = 30;
borrow_lim  = 0;
agrid_par   = 0.4; %1 for linear, 0 for L-shaped

% computation
max_iter    = 1000;
tol_iter    = 1.0e-6;
Nsim        = 50000;
Tsim        = 500;

%% OPTIONS
Display     = 1;
DoSimulate  = 1;
MakePlots   = 1;

% which function to interpolation
InterpCon = 0;
InterpEMUC = 1;
