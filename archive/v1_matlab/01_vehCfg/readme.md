## Vehicle Trajectory Configuration Tool

<ins>"vehCfgManualTool.m"</ins> : Manually generate trajectories that can be used in "../02_v2lcDataGen". This script takes waypoints on a trajectory as inputs and interpolates between the waypoints with resolution depending on sampling constraints. There are two sampling periods in our simulator: vehicle simulation sampling period and VLC simulation sampling period. The sampling period here denotes the vehicle simulation sampling period. 

<ins>"vehCfgManualTool_3lane.m"</ins> : Same tool, just for the SM4 static 3-lane operational range characterization scenario. Generates static location grid for that simulation rather than a dynamic trajectory.
