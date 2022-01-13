import numpy as np

filename = "/Users/yifanc/Research/python_code/GP-PDEs-time-stepping/HighDimeqn/logs/NonVarLinElliptic/dim4_kernelinv_quadratics/alpha1.0m3sigma-scale0.25_Ndomain1000_Nbd200_nugget1e-10_freqa1.0_frequ1.0_numexp5_0112_211331.log.npz"

x = np.load(filename, allow_pickle=True)
print(x['L2'])
print(x['Linf'])