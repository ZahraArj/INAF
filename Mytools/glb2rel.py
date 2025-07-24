import numpy as np
from Mytools.Myinv import rot_inv


glb_odo = np.genfromtxt("/home/zara/Desktop/Base_Code_LO/results/supervised_all/result_200_screw/output_trajectory_parts/seq_00.txt")

for idx in range(5, 1000):
    # print('geo', i - 1, i)

    if idx == 0:
        gt_1 = np.array([1, 0, 0, 0,
                         0, 1, 0, 0,
                         0, 0, 1, 0])
        gt_2 = glb_odo[idx]
    else:
        gt_1 = glb_odo[idx - 1]
        gt_2 = glb_odo[idx]

    gt_T1 = np.array([[gt_1[1], gt_1[2], gt_1[3], gt_1[4]],
                      [gt_1[5], gt_1[6], gt_1[7], gt_1[8]],
                      [gt_1[9], gt_1[10], gt_1[11], gt_1[12]],
                      [0, 0, 0, 1]])
    gt_T2 = np.array([[gt_2[1], gt_2[2], gt_2[3], gt_2[4]],
                      [gt_2[5], gt_2[6], gt_2[7], gt_2[8]],
                      [gt_2[9], gt_2[10], gt_2[11], gt_2[12]],
                      [0, 0, 0, 1]])

    temp = np.array([[0, 0, 1, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]])

    # global_transform1 = temp @ gt_T1 @ self.loader.calib.T_cam0_velo
    # global_transform2 = temp @ gt_T2 @ self.loader.calib.T_cam0_velo

    global_transform1 = gt_T1
    global_transform2 = gt_T2

    pose = [gt_2[3], gt_2[7], gt_2[11]]

    # print (global_transform1)
    # print (global_transform2)

    # global_transform1_inv = np.linalg.inv(global_transform1)
    # T = np.dot(global_transform1_inv, global_transform2)

    global_transform1_inv = rot_inv(global_transform1)
    T = np.dot(global_transform1_inv, global_transform2)

    # T = np.linalg.solve(global_transform1, global_transform2)

    # T = global_transform1

    return T, global_transform2, pose

filepath_rel = os.path.join(mother_folder, 'results', 'output_trajectory_parts', 'seq_' + file_id + '_rel.txt')