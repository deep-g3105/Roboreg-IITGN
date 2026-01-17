import numpy as np

T_c_f = np.array([[ 0.3385,  0.1580, -0.9276, -0.1474],
        [ 0.1282, -0.9843, -0.1209, -0.4396],
        [-0.9322, -0.0780, -0.3535,  1.2247],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])

T_opt = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])

T_c_h = np.array([[-0.1373,  0.9720,  0.1909, -0.3643],
        [ 0.3415,  0.2273, -0.9120, -0.1629],
        [-0.9298, -0.0600, -0.3631,  1.2174],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])
# Matrix multiplication
T_c_h_opt = T_c_h 
T_c_f_opt =  T_c_f 
# or: C = np.matmul(T_c_f, T_opt)

# print(T_c_h_opt)



T_inv_h = np.linalg.inv(T_c_h_opt)

# print(T_inv_h)
T_h_f = T_inv_h @ T_c_f_opt
print(T_h_f)