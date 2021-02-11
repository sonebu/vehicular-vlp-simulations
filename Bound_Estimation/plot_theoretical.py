import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from matfile_read import load_mat


def deviation_from_actual_value(array, actual_val):
    return np.sqrt(np.mean(abs(array - actual_val) ** 2, axis=0))

def main():
    dir = '../GUI_data/100_point_'
    files = glob(dir + '*/3/')
    x_pose, y_pose = [], []
    x_becha, y_becha = [], []
    x_roberts, y_roberts = [], []
    dp = 10
    print(len(files))

    for folder_name in files:
        x_pose.append(np.loadtxt(folder_name+'x_pose.txt', delimiter=','))
        y_pose.append(np.loadtxt(folder_name+'y_pose.txt', delimiter=','))
        x_roberts.append(np.loadtxt(folder_name+'x_roberts.txt', delimiter=','))
        y_roberts.append(np.loadtxt(folder_name+'y_roberts.txt', delimiter=','))
        x_becha.append(np.loadtxt(folder_name+'x_becha.txt', delimiter=','))
        y_becha.append(np.loadtxt(folder_name+'y_becha.txt', delimiter=','))
    # print(np.shape(np.asarray(y_pose)))
    data = load_mat('../SimulationData/v2lcRun_sm3_comparisonSoA.mat')
    time = data['vehicle']['t']['values'][::dp]
    tx1_x = data['vehicle']['target_relative']['tx1_qrx4']['y'][::dp]
    tx1_y = data['vehicle']['target_relative']['tx1_qrx4']['x'][::dp]
    tx2_x = data['vehicle']['target_relative']['tx2_qrx3']['y'][::dp]
    tx2_y = data['vehicle']['target_relative']['tx2_qrx3']['x'][::dp]

    pose_x1_crlb = np.loadtxt('Data/aoa/crlb_x1.txt', delimiter=',')
    becha_x1_crlb = np.loadtxt('Data/rtof/crlb_x1.txt', delimiter=',')
    pose_x2_crlb = np.loadtxt('Data/aoa/crlb_x2.txt', delimiter=',')
    becha_x2_crlb = np.loadtxt('Data/rtof/crlb_x2.txt', delimiter=',')
    roberts_x_crlb = np.loadtxt('Data/tdoa/crlb_x.txt', delimiter=',')

    pose_y1_crlb = np.loadtxt('Data/aoa/crlb_y1.txt', delimiter=',')
    becha_y1_crlb = np.loadtxt('Data/rtof/crlb_y1.txt', delimiter=',')
    pose_y2_crlb = np.loadtxt('Data/aoa/crlb_y2.txt', delimiter=',')
    becha_y2_crlb = np.loadtxt('Data/rtof/crlb_y2.txt', delimiter=',')
    roberts_y_crlb = np.loadtxt('Data/tdoa/crlb_y.txt', delimiter=',')

    plt.close("all")
    plot1 = plt.figure(1)
    becha_x1, = plt.plot(time, becha_x1_crlb)
    soner_x1, = plt.plot(time, pose_x1_crlb)
    ten_cm_line, = plt.plot(time, 0.1*np.ones(100),'--')
    roberts_x1, = plt.plot(time, roberts_x_crlb)
    plt.ylabel('Standard Deviation (m)')
    plt.xlabel('Time (s)')
    plt.title('CRLB for x1')
    plt.legend([becha_x1, soner_x1, roberts_x1, ten_cm_line], ['RToF', 'AoA', 'TDoA', '10 cm line'])
    plt.ylim(1e-5,10)
    plt.yscale('log')
    plt.savefig('crlb_x1.png')

    plot2 = plt.figure(2)
    becha_y1, = plt.plot(time, becha_y1_crlb)
    soner_y1, = plt.plot(time, pose_y1_crlb)
    ten_cm_line, = plt.plot(time, 0.1*np.ones(100),'--')
    roberts_y1, = plt.plot(time, roberts_y_crlb)
    plt.ylabel('Standard Deviation (m)')
    plt.xlabel('Time (s)')
    plt.title('CRLB for y1')
    plt.legend([becha_y1, soner_y1, roberts_y1, ten_cm_line], ['RToF', 'AoA', 'TDoA', '10 cm line'])
    plt.ylim(1e-5,10)
    plt.yscale('log')
    plt.savefig('crlb_y1.png')

    plot3 = plt.figure(3)
    becha_x2, = plt.plot(time, becha_x2_crlb)
    soner_x2, = plt.plot(time, pose_x2_crlb)
    ten_cm_line, = plt.plot(time, 0.1*np.ones(100),'--')
    plt.ylabel('Standard Deviation (m)')
    plt.xlabel('Time (s)')
    plt.title('CRLB for x2')
    plt.legend([becha_x2, soner_x2, ten_cm_line], ['RToF', 'AoA', '10 cm line'])
    plt.ylim(1e-5,10)
    plt.yscale('log')
    plt.savefig('crlb_x2.png')

    plot4 = plt.figure(4)
    becha_y2, = plt.plot(time, becha_y2_crlb)
    soner_y2, = plt.plot(time, pose_y2_crlb)
    ten_cm_line, = plt.plot(time, 0.1*np.ones(100),'--')
    plt.ylabel('Standard Deviation (m)')
    plt.xlabel('Time (s)')
    plt.title('CRLB for y2')
    plt.legend([becha_y2, soner_y2, ten_cm_line], ['RToF', 'AoA', '10 cm line'])
    plt.ylim(1e-5,10)
    plt.yscale('log')
    plt.savefig('crlb_y2.png')

    x1_becha, x2_becha = np.asarray(x_becha)[:,:,0], np.asarray(x_becha)[:,:,1]
    y1_becha, y2_becha = np.asarray(y_becha)[:,:,0], np.asarray(y_becha)[:,:,1]
    print(np.shape(x1_becha))
    print(np.shape(x2_becha))
    print(np.shape(y1_becha))
    print(np.shape(y2_becha))

    plot5 = plt.figure(5)
    th_becha_x1, = plt.plot(time, becha_x1_crlb, '--')
    th_becha_x2, = plt.plot(time, becha_x2_crlb, '--')
    th_becha_y1, = plt.plot(time, becha_y1_crlb, '--')
    th_becha_y2, = plt.plot(time, becha_y2_crlb, '--')
    # sim_becha_x1, = plt.plot(time[0:i + 1], abs(tx1_x + x1_becha)[0:i + 1])
    sim_becha_x1, = plt.plot(time, deviation_from_actual_value(x1_becha, -tx1_x))
    # sim_becha_x2, = plt.plot(time[0:i + 1], abs(tx2_x + x2_becha)[0:i + 1])
    sim_becha_x2, = plt.plot(time, deviation_from_actual_value(x2_becha, -tx2_x))
    # sim_becha_y1, = plt.plot(time[0:i + 1], abs(tx1_y - y1_becha)[0:i + 1])
    sim_becha_y1, = plt.plot(time, deviation_from_actual_value(y1_becha, tx1_y))
    # sim_becha_y2, = plt.plot(time[0:i + 1], abs(tx2_y - y2_becha)[0:i + 1])
    sim_becha_y2, = plt.plot(time, deviation_from_actual_value(y2_becha, tx2_y))
    plt.ylabel('Error (m)')
    plt.xlabel('Time (s)')
    plt.title('CRLB vs. Simulation Results for RToF')
    # plt.legend([th_becha_x1, th_becha_x2, th_becha_y1, th_becha_y2, sim_becha_x1 , sim_becha_x2, sim_becha_y1,
    #             sim_becha_y2], ['x1 (theoretical)', 'x2 (theoretical)', 'y1 (theoretical)', 'y2 (theoretical)',
    #                             'x1 (simulation)', 'x2 (simulation)', 'y1 (simulation)', 'y2 (simulation)'],
    #            ncol=4,loc=3)
    # plt.ylim(1e-5,2)
    # plt.yscale('log')
    plt.savefig('crlb_becha_lin.png')
    plt.yscale('log')
    plt.savefig('crlb_becha_log.png')

    x1_roberts, x2_roberts = np.asarray(x_roberts)[:,:,0], np.asarray(x_roberts)[:,:,1]
    y1_roberts, y2_roberts = np.asarray(y_roberts)[:,:,0], np.asarray(y_roberts)[:,:,1]

    plot6 = plt.figure(6)
    th_roberts_x1, = plt.plot(time, roberts_x_crlb, '--')
    th_roberts_y1, = plt.plot(time, roberts_y_crlb, '--')
    #xr_mask = np.isfinite(deviation_from_actual_value(x1_roberts, tx1_x))
    #yr_mask = np.isfinite(deviation_from_actual_value(y1_roberts, tx1_y))
    sim_roberts_x1, = plt.plot(time, deviation_from_actual_value(x1_roberts, -tx1_x))
    #sim_roberts_x1, = plt.plot(time[0:i + 1][xr_mask], deviation_from_actual_value(x1_roberts, -tx1_x)[xr_mask])
    sim_roberts_y1, = plt.plot(time, deviation_from_actual_value(y1_roberts, tx1_y))
    #sim_roberts_y1, = plt.plot(time[0:i + 1][yr_mask], deviation_from_actual_value(y1_roberts, tx1_y)[yr_mask])
    plt.ylabel('Error (m)')
    plt.xlabel('Time (s)')
    plt.title('CRLB vs. Simulation Results for TDoA')
    plt.legend([th_roberts_x1, th_roberts_y1, sim_roberts_x1 , sim_roberts_y1],
               ['x1 (theoretical)', 'y1 (theoretical)', 'x1 (simulation)', 'y1 (simulation)'],ncol=2,loc=3)
    plt.ylim(1e-5,2)
    plt.yscale('log')
    plt.savefig('crlb_roberts.png')

    x1_soner, x2_soner = np.asarray(x_pose)[:,:,0], np.asarray(x_pose)[:,:,1]
    y1_soner, y2_soner = np.asarray(y_pose)[:,:,0], np.asarray(y_pose)[:,:,1]

    plot7 = plt.figure(7)
    th_soner_x1, = plt.plot(time, pose_x1_crlb, '--')
    th_soner_x2, = plt.plot(time, pose_x2_crlb, '--')
    th_soner_y1, = plt.plot(time, pose_y1_crlb, '--')
    th_soner_y2, = plt.plot(time, pose_y2_crlb, '--')
    # sim_soner_x1, = plt.plot(time[0:i + 1], abs(tx1_x + x1_soner)[0:i + 1])
    sim_soner_x1, = plt.plot(time, deviation_from_actual_value(x1_soner, -tx1_x))
    # sim_soner_x2, = plt.plot(time[0:i + 1], abs(tx2_x + x2_soner)[0:i + 1])
    sim_soner_x2, = plt.plot(time, deviation_from_actual_value(x2_soner, -tx2_x))
    # sim_soner_y1, = plt.plot(time[0:i + 1], abs(tx1_y - y1_soner)[0:i + 1])
    sim_soner_y1, = plt.plot(time, deviation_from_actual_value(y1_soner, tx1_y))
    # sim_soner_y2, = plt.plot(time[0:i + 1], abs(tx2_y - y2_soner)[0:i + 1])
    sim_soner_y2, = plt.plot(time, deviation_from_actual_value(y2_soner, tx2_y))
    plt.ylabel('Error (m)')
    plt.xlabel('Time (s)')
    plt.title('CRLB vs. Simulation Results for AoA')
    plt.legend([th_soner_x1, th_soner_x2, th_soner_y1, th_soner_y2, sim_soner_x1 , sim_soner_x2, sim_soner_y1,
                sim_soner_y2], ['x1 (theoretical)', 'x2 (theoretical)', 'y1 (theoretical)', 'y2 (theoretical)',
                                'x1 (simulation)', 'x2 (simulation)', 'y1 (simulation)', 'y2 (simulation)'],
               ncol=4,loc=3)
    plt.ylim(1e-5,2)
    plt.yscale('log')
    plt.savefig('crlb_soner.png')
    plt.show()

if __name__ == "__main__":
    main()