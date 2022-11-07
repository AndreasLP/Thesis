import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import signal

# Get data
simulation = np.fromfile("ode_output.bin").reshape((-1,90))

values = np.row_stack([np.array([True]),(np.diff(simulation[:,0]) > 0).reshape(-1,1)]).ravel()
interpolator = interp1d(simulation[values,0], simulation[values,1:], axis=0, kind='linear', fill_value="extrapolate")
interpolated = np.column_stack(( np.arange(0, 140910) / 2, interpolator( np.arange(50, 140910, 0.5) ) ))

df = pd.DataFrame(interpolated, columns=["position", "velocity", "curvature", "tau", 
"u[1]", "u[2]", "u[3]", "u[4]", "u[5]", "u[6]", "u[7]", "u[8]", "u[9]", 
"u[10]", "u[11]", "u[12]", "u[13]", "u[14]", "u[15]", "u[16]", "u[17]", "u[18]", "u[19]", 
"u[20]", "u[21]", "u[22]", "u[23]", "u[24]", "u[25]", "u[26]", "u[27]", "u[28]", "u[29]", 
"u[30]", "du[2]", "du[4]", "du[6]", "du[8]", "du[10]", "du[12]", "du[14]", "du[16]", "du[18]", 
"du[20]", "du[22]", "du[24]", "du[26]", "du[28]", "du[29]", "du[30]", 
"irreg[1,1]", "irreg[2,1]", "irreg[3,1]", "irreg[4,1]", 
"irreg[1,2]", "irreg[2,2]", "irreg[3,2]", "irreg[4,2]", 
"data[1,1]", "data[1,2]", "data[1,3]", "data[1,4]", "data[1,5]", "data[1,6]", "data[1,7]", "data[1,8]", 
"data[2,1]", "data[2,2]", "data[2,3]", "data[2,4]", "data[2,5]", "data[2,6]", "data[2,7]", "data[2,8]", 
"data[3,1]", "data[3,2]", "data[3,3]", "data[3,4]", "data[3,5]", "data[3,6]", "data[3,7]", "data[3,8]", 
"data[4,1]", "data[4,2]", "data[4,3]", "data[4,4]", "data[4,5]", "data[4,6]", "data[4,7]", "data[4,8]"])

position = df.loc[:,'position']
irreg = df.loc[:,[c for c in df.columns if c.startswith("irreg")]]
ode_accelerations = df.loc[:,[c for c in df.columns if c.startswith("du")]]

irreg.columns = [
    "fw left lateral", "fw right lateral", "rw left lateral", "rw right lateral", 
    "fw left vertical", "fw right vertical", "rw left vertical", "rw right vertical"
]
y_unfiltered = irreg.loc[:,["fw left lateral", "fw right lateral", "fw left vertical", "fw right vertical"]]*1000
y_unfiltered.columns = ['LivLongSx [mm]', 'LivLongDx [mm]', 'AllinSx [mm]', 'AllinDx [mm]']

sosD1 = signal.butter(10, [1/25,  1/3 ], 'bandpass', fs=2, output='sos')
sosD2 = signal.butter(10, [1/70,  1/25], 'bandpass', fs=2, output='sos')
sosD3 = signal.butter(10, [1/200, 1/70], 'bandpass', fs=2, output='sos')

filteredD1 = signal.sosfilt(sosD1, y_unfiltered.values)
filteredD2 = signal.sosfilt(sosD2, y_unfiltered.values)
filteredD3 = signal.sosfilt(sosD3, y_unfiltered.values)
y_filtered = pd.DataFrame(np.column_stack((position, filteredD1, filteredD2, filteredD3)), columns=["position", 
    "LivLongSxD1 [mm]", "LivLongDxD1 [mm]", "AllinSxD1 [mm]", "AllinDxD1 [mm]",
    "LivLongSxD2 [mm]", "LivLongDxD2 [mm]", "AllinSxD2 [mm]", "AllinDxD2 [mm]",
    "LivLongSxD3 [mm]", "LivLongDxD3 [mm]", "AllinSxD3 [mm]", "AllinDxD3 [mm]",
]).set_index("position")

ode_accelerations.columns = [
	"FW lateral acceleration", "FW yaw circular acceleration", "RW lateral acceleration", "RW yaw circular acceleration",
	"bogie frame lateral acceleration", "bogie frame yaw circular acceleration", "bogie frame roll circular acceleration",
	"Car body roll circular acceleration", "FW vertical acceleration", "RW vertical acceleration",
	"FW roll circular acceleration", "RW roll circular acceleration", "bogie frame vertical acceleration",
	"bogie frame pitch circular acceleration", "Rolling constraint p.beta1 (FW)", "Rolling constraint p.beta2 (RW)"
]

# AB: Axel box
r_AB = 0.965

AB_Y_1 = ode_accelerations.loc[:,"FW lateral acceleration"] 
AB_Y_2 = ode_accelerations.loc[:,"FW lateral acceleration"]
AB_Y_3 = ode_accelerations.loc[:,"RW lateral acceleration"]
AB_Y_4 = ode_accelerations.loc[:,"RW lateral acceleration"]

AB_Z_1 = ode_accelerations.loc[:,"FW vertical acceleration"] + ode_accelerations.loc[:,"FW roll circular acceleration"] * r_AB
AB_Z_2 = ode_accelerations.loc[:,"FW vertical acceleration"] - ode_accelerations.loc[:,"FW roll circular acceleration"] * r_AB
AB_Z_3 = ode_accelerations.loc[:,"RW vertical acceleration"] + ode_accelerations.loc[:,"RW roll circular acceleration"] * r_AB
AB_Z_4 = ode_accelerations.loc[:,"RW vertical acceleration"] - ode_accelerations.loc[:,"RW roll circular acceleration"] * r_AB

l_B = 1.500

B_Y_1 = ode_accelerations.loc[:,"bogie frame lateral acceleration"] + ode_accelerations.loc[:,"bogie frame yaw circular acceleration"]*l_B
B_Y_2 = ode_accelerations.loc[:,"bogie frame lateral acceleration"] + ode_accelerations.loc[:,"bogie frame yaw circular acceleration"]*l_B
B_Y_3 = ode_accelerations.loc[:,"bogie frame lateral acceleration"] - ode_accelerations.loc[:,"bogie frame yaw circular acceleration"]*l_B
B_Y_4 = ode_accelerations.loc[:,"bogie frame lateral acceleration"] - ode_accelerations.loc[:,"bogie frame yaw circular acceleration"]*l_B

B_Z_1 = ode_accelerations.loc[:,"bogie frame vertical acceleration"] - ode_accelerations.loc[:,"bogie frame pitch circular acceleration"]*l_B
B_Z_2 = ode_accelerations.loc[:,"bogie frame vertical acceleration"] - ode_accelerations.loc[:,"bogie frame pitch circular acceleration"]*l_B
B_Z_3 = ode_accelerations.loc[:,"bogie frame vertical acceleration"] + ode_accelerations.loc[:,"bogie frame pitch circular acceleration"]*l_B
B_Z_4 = ode_accelerations.loc[:,"bogie frame vertical acceleration"] + ode_accelerations.loc[:,"bogie frame pitch circular acceleration"]*l_B

CB_Y_1 = ode_accelerations.loc[:, "Car body roll circular acceleration"]
CB_Y_2 = ode_accelerations.loc[:, "Car body roll circular acceleration"]

CB_Z_1 = 0*ode_accelerations.loc[:, "Car body roll circular acceleration"]
CB_Z_2 = 0*ode_accelerations.loc[:, "Car body roll circular acceleration"]

# Save results to a relevant output file
