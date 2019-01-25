# DADAM: A Consensus-based Distributed Adaptive Gradient Method for Online Optimization
The MATLAB code is built on top of the SGDLibrary. You can find the latest vesion here: https://github.com/hiroyuki-kasai/SGDLibrary 

# First to do
Run run_me_first_to_add_libs_.m for path configurations.

%% First run the setup script
% Add folders to path.

addpath(pwd);

cd SGDLibrary-master/;
addpath(genpath(pwd));
cd ..;

cd DADAM-master/;
addpath(genpath(pwd));
cd ..;
