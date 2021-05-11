## VLC Configuration Tool

<ins>"vlcCfgTool_script.m"</ins> : Configure the VLC transmitter and receiver and load/save the configuration to a "data/vlcCfg_<>.mat" file for later use by "../02_v2lcDataGen". 

The novel receiver architecture, named QRX, consists of a quadrant-photodiode and a hemispherical lens. The QRX configuration parameters determine a certain mapping from the Spot Position (determines power on each quadrant) to \theta, which is the angle-of-arrival; this mapping is plotted by the script after a run. The mapping is calculated by ray optics relations. The resolution is controlled by the simRes parameters. The mapping is related to f_QRX in the submitted article, the inverse of which is used for estimating angle-of-arrival.