import os

# os.system("CUDA_VISIBLE_DEVICES=0 nohup python accuracy_wrt_d_fix_auN.py --dim_low 3 --dim_high 20 --N_domain 500 --N_boundary 100 --num_exp 100 --freq_a 1.0 --freq_u 4.0 &")

# os.system("CUDA_VISIBLE_DEVICES=1 nohup python accuracy_wrt_d_fix_auN.py --dim_low 3 --dim_high 20 --N_domain 1000 --N_boundary 200 --num_exp 100 --freq_a 1.0 --freq_u 4.0 &")

# os.system("CUDA_VISIBLE_DEVICES=2 nohup python accuracy_wrt_d_fix_auN.py --dim_low 3 --dim_high 20 --N_domain 2000 --N_boundary 400 --num_exp 100 --freq_a 1.0 --freq_u 4.0 &")

# os.system("CUDA_VISIBLE_DEVICES=3 nohup python accuracy_wrt_d_fix_auN.py --dim_low 3 --dim_high 20 --N_domain 4000 --N_boundary 800 --num_exp 100 --freq_a 1.0 --freq_u 4.0 &")

os.system("CUDA_VISIBLE_DEVICES=0 nohup python accuracy_wrt_d_fix_auN.py --dim_low 3 --dim_high 20 --N_domain 8000 --N_boundary 1600 --num_exp 100 --freq_a 1.0 --freq_u 4.0 &")