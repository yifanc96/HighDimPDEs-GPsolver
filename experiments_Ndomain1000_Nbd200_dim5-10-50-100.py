import os

os.system("CUDA_VISIBLE_DEVICES=0 nohup python NonLinVariableElliptic_GPsolver.py --dim 5 --N_domain 1000 --N_boundary 200 --num_exp 100 --freq_a 1.0 --freq_u 4.0 &")

os.system("CUDA_VISIBLE_DEVICES=1 nohup python NonLinVariableElliptic_GPsolver.py --dim 10 --N_domain 1000 --N_boundary 200 --num_exp 100 --freq_a 1.0 --freq_u 4.0 &")

os.system("CUDA_VISIBLE_DEVICES=2 nohup python NonLinVariableElliptic_GPsolver.py --dim 50 --N_domain 1000 --N_boundary 200 --num_exp 100 --freq_a 1.0 --freq_u 4.0 &")

os.system("CUDA_VISIBLE_DEVICES=3 nohup python NonLinVariableElliptic_GPsolver.py --dim 100 --N_domain 1000 --N_boundary 200 --num_exp 100 --freq_a 1.0 --freq_u 4.0 &")

