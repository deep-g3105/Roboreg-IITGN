import numpy as np

T_c_f = np.array([[-0.2857,  0.1682, -0.9434,  0.6933],
        [ 0.8751, -0.3554, -0.3284, -1.7426],
        [-0.3905, -0.9195, -0.0456,  0.6695],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])

T_opt = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])

T_c_h = np.array([[-0.8627,  0.3074,  0.4016,  0.8729],
        [-0.3599,  0.1847, -0.9145,  0.7893],
        [-0.3553, -0.9335, -0.0487,  0.6163],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])
# Matrix multiplication
T_c_h_opt = T_c_h @ T_opt
T_c_f_opt =  T_c_f @ T_opt
# or: C = np.matmul(T_c_f, T_opt)

# print(T_c_h_opt)



T_inv_h = np.linalg.inv(T_c_h_opt)

# print(T_inv_h)
T_h_f = T_inv_h @ T_c_f_opt
print(T_h_f)