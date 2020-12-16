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

#data = load_mat('SimulationData/v2lcRun_sm3_comparisonSoA.mat')
#rec_func(data, 0)
#print(data['channel']['qrx1']['delay']['tx1'])
#print(len(data['channel']['qrx1']['delay']['tx1']))
#print(data['channel']['qrx1']['delay']['tx1'][-1])
#print(data['channel']['qrx1']['delay']['tx2'])
#p_r_factor = data['qrx']['tia']['shot_P_r_factor']
#i_bg_factor = data['qrx']['tia']['shot_I_bg_factor']

#print(p_r_factor)
#print(i_bg_factor)