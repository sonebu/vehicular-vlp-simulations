from mat4py import loadmat
from scipy.io import loadmat, matlab
import numpy as np


def load_mat(filename):
    """
    This function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """

    def _check_vars(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
            elif isinstance(d[key], np.ndarray):
                d[key] = _toarray(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _toarray(elem)
            else:
                d[strg] = elem
        return d

    def _toarray(ndarray):
        """
        A recursive function which constructs ndarray from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        if ndarray.dtype != 'float64':
            elem_list = []
            for sub_elem in ndarray:
                if isinstance(sub_elem, matlab.mio5_params.mat_struct):
                    elem_list.append(_todict(sub_elem))
                elif isinstance(sub_elem, np.ndarray):
                    elem_list.append(_toarray(sub_elem))
                else:
                    elem_list.append(sub_elem)
            return np.array(elem_list)
        else:
            return ndarray

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_vars(data)


def rec_func(data, n):
    for k in data.keys():
        str = ""
        for i in range(n):
            str += "\t"
        print(str, k)
        if isinstance(data[k], dict):
            rec_func(data[k], n + 1)

data = load_mat('../SimulationData/v2lcRun_sm3_comparisonSoA.mat')
#data = load_mat('VLP_methods/aoa_transfer_function.mat')
rec_func(data, 0)
#print(data['breaks'])
#print(data['coefs'])
#print('ego vehicle front x: ',data['vehicle']['ego']['front_center']['x'][0], ' y:',data['vehicle']['ego']['front_center']['y'][0])
print('ego vehicle qrx1 x:', data['vehicle']['ego']['tx4_qrx1']['x'][0], ' y:', data['vehicle']['ego']['tx4_qrx1']['y'][0])
print('ego vehicle qrx2 x:', data['vehicle']['ego']['tx3_qrx2']['x'][0], ' y:', data['vehicle']['ego']['tx3_qrx2']['y'][0])


print('target vehicle tx1 x:', data['vehicle']['target']['tx1_qrx4']['x'][0], ' y:', data['vehicle']['target']['tx1_qrx4']['y'][0])
print('target vehicle relative tx1 x:', data['vehicle']['target_relative']['tx1_qrx4']['x'][0], 'y:',data['vehicle']['target_relative']['tx1_qrx4']['y'][0])

print('target vehicle tx2 x:',data['vehicle']['target']['tx2_qrx3']['x'][0], 'y:',data['vehicle']['target']['tx2_qrx3']['y'][0])
print('target vehicle relative tx2 x:',data['vehicle']['target_relative']['tx2_qrx3']['x'][0] , 'y:' , data['vehicle']['target_relative']['tx2_qrx3']['y'][0] )
print(data['vehicle']['target']['tx2_qrx3']['x'][0]-data['vehicle']['ego']['tx4_qrx1']['x'][0])
print(data['vehicle']['target']['tx2_qrx3']['y'][0]-data['vehicle']['ego']['tx4_qrx1']['y'][0])

#print('ego vehicle front x: ',data['vehicle']['ego']['front_center']['x'][0], ' y:',data['vehicle']['ego']['front_center']['y'][1])
#print('ego vehicle qrx1 x:', data['vehicle']['ego']['tx4_qrx1']['x'][0], ' y:', data['vehicle']['ego']['tx4_qrx1']['y'][1])
#print('ego vehicle qrx2 x:', data['vehicle']['ego']['tx3_qrx2']['x'][0], ' y:', data['vehicle']['ego']['tx3_qrx2']['y'][1])

#print('target vehicle rear x:', data['vehicle']['target']['rear_center']['x'][0], ' y:', data['vehicle']['target']['rear_center']['y'][1])
#print('target vehicle relative rear x:', data['vehicle']['target_relative']['rear_center']['x'][0], 'y:',data['vehicle']['target_relative']['rear_center']['y'][1])

#print('target vehicle tx1 x:', data['vehicle']['target']['tx1_qrx4']['x'][0], ' y:', data['vehicle']['target']['tx1_qrx4']['y'][1])
#print('target vehicle relative tx1 x:', data['vehicle']['target_relative']['tx1_qrx4']['x'][0], 'y:',data['vehicle']['target_relative']['tx1_qrx4']['y'][1])

#print('target vehicle tx2 x:',data['vehicle']['target']['tx2_qrx3']['x'][0], 'y:',data['vehicle']['target']['tx2_qrx3']['y'][1])
#print('target vehicle relative tx2 x:',data['vehicle']['target_relative']['tx2_qrx3']['x'][0] , 'y:' , data['vehicle']['target_relative']['tx2_qrx3']['y'][1] )

#print((data['vehicle']['target']['tx1_qrx4']['x'] - data['vehicle']['ego']['tx4_qrx1']['x']) == data['vehicle']['target_relative']['tx1_qrx4']['x'])
#print((data['vehicle']['target']['tx1_qrx4']['y'] - data['vehicle']['ego']['tx4_qrx1']['y']) == data['vehicle']['target_relative']['tx1_qrx4']['y'])

#print((data['vehicle']['target']['tx2_qrx3']['x'] - data['vehicle']['ego']['tx4_qrx1']['x']) == data['vehicle']['target_relative']['tx2_qrx3']['x'])
#print((data['vehicle']['target']['tx2_qrx3']['y'] - data['vehicle']['ego']['tx4_qrx1']['y']) == data['vehicle']['target_relative']['tx2_qrx3']['y'])

#print(data['channel']['qrx1']['power']['tx1']['A'][1])
#print(data['channel']['qrx1']['power']['tx1']['B'][1])
#print(data['channel']['qrx1']['power']['tx1']['C'][1])
#print(data['channel']['qrx1']['power']['tx1']['D'][1])
#power = np.array([data['channel']['qrx1']['power']['tx1']['A'], data['channel']['qrx1']['power']['tx1']['B'],
#                 data['channel']['qrx1']['power']['tx1']['C'], data['channel']['qrx1']['power']['tx1']['D']])
#print(power.shape)
#print(power[:, 1])
#new_arr = np.array([[power[:, 0], power[:, 1]], [power[:, 2], power[:, 3]]])
#print(new_arr.shape)
#print(new_arr[0][1])
#print(new_arr[0][1][1])
#print(np.sum(new_arr[0][1]))
#print(data['qrx']['tia'])
#print(data['qrx']['tia']['shot_P_r_factor'])
#print(data['qrx']['tia']['shot_I_bg_factor'])
#print(data['qrx']['tia']['thermal_factor1'])
#print(data['qrx']['tia']['thermal_factor2'])
#print(data['channel']['qrx1']['delay']['tx1'])
#print(len(data['channel']['qrx1']['delay']['tx1']))
#print(data['channel']['qrx1']['delay']['tx1'][-1])
#print(data['channel']['qrx1']['delay']['tx2'])
#p_r_factor = data['qrx']['tia']['shot_P_r_factor']
#i_bg_factor = data['qrx']['tia']['shot_I_bg_factor']

#print(p_r_factor)
#print(i_bg_factor)