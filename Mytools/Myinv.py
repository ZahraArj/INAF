import numpy as np


def rot_inv(T):
    invT = np.zeros((4, 4))
    inv_Rot = np.array([[T[0, 0], T[1, 0], T[2, 0]],
                        [T[0, 1], T[1, 1], T[2, 1]],
                        [T[0, 2], T[1, 2], T[2, 2]],
                        [0, 0, 0]])
    invT[0:4, 0:3] = inv_Rot
    transl = - np.dot(inv_Rot[0:3, :], T[0:3, 3])
    invT[0:3, 3] = transl
    invT[3, 3] = 1

    return invT
