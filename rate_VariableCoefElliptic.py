import jax.numpy as jnp
from jax import grad, vmap, hessian

from jax.config import config
config.update("jax_enable_x64", True)

# numpy
import numpy as onp
from numpy import random 

import argparse
import logging
import datetime
from time import time
import os

# solving -grad(a*grad u) + alpha u^m = f
def get_parser():
    parser = argparse.ArgumentParser(description='NonLinElliptic equation GP solver')
    parser.add_argument("--freq_a", type=float, default = 1.0)
    parser.add_argument("--alpha", type=float, default = 1.0)
    parser.add_argument("--m", type = int, default = 3)
    parser.add_argument("--dim", type = int, default = 2)
    parser.add_argument("--kernel", type=str, default="Matern_7half", choices=["gaussian","inv_quadratics","Matern_3half","Matern_5half","Matern_7half","Matern_9half","Matern_11half"])
    parser.add_argument("--sigma-scale", type = float, default = 0.25)
    # sigma = args.sigma-scale*sqrt(dim)
    
    parser.add_argument("--nugget", type = float, default = 1e-10)
    parser.add_argument("--GNsteps", type = int, default = 6)
    parser.add_argument("--logroot", type=str, default='./logs/')
    parser.add_argument("--randomseed", type=int, default=1)
    parser.add_argument("--num_exp", type=int, default=1)
    args = parser.parse_args()    
    return args

def get_GNkernel_train(x,y,wx0,wx1,wxg,wy0,wy1,wyg,d,sigma):
    # wx0 * delta_x + wxg * nabla delta_x + wx1 * Delta delta_x 
    return wx0*wy0*kappa(x,y,d,sigma) + wx0*wy1*Delta_y_kappa(x,y,d,sigma) + wy0*wx1*Delta_x_kappa(x,y,d,sigma) + wx1*wy1*Delta_x_Delta_y_kappa(x,y,d,sigma) + wx0*D_wy_kappa(x,y,d,sigma,wyg) + wy0*D_wx_kappa(x,y,d,sigma,wxg) + wx1*Delta_x_D_wy_kappa(x,y,d,sigma,wyg) + wy1*D_wx_Delta_y_kappa(x,y,d,sigma,wxg) + D_wx_D_wy_kappa(x,y,d,sigma,wxg,wyg)

def get_GNkernel_train_boundary(x,y,wy0,wy1,wyg,d,sigma):
    return wy0*kappa(x,y,d,sigma) + wy1*Delta_y_kappa(x,y,d,sigma) + D_wy_kappa(x,y,d,sigma,wyg)

def get_GNkernel_val_predict(x,y,wy0,wy1,wyg,d,sigma):
    return wy0*kappa(x,y,d,sigma) + wy1*Delta_y_kappa(x,y,d,sigma) + D_wy_kappa(x,y,d,sigma,wyg)

def get_GNkernel_val_predict_Delta(x,y,wy0,wy1,wyg,d,sigma):
    return wy0*Delta_x_kappa(x,y,d,sigma) + wy1*Delta_x_Delta_y_kappa(x,y,d,sigma) + Delta_x_D_wy_kappa(x,y,d,sigma,wyg)


def assembly_Theta(X_domain, X_boundary, w0, w1, wg, sigma):
    # X_domain, dim: N_domain*d; 
    # w0 col vec: coefs of Diracs, dim: N_domain; 
    # w1 coefs of Laplacians, dim: N_domain
    
    N_domain,d = onp.shape(X_domain)
    N_boundary,_ = onp.shape(X_boundary)
    Theta = onp.zeros((N_domain+N_boundary,N_domain+N_boundary))
    
    XdXd0 = onp.reshape(onp.tile(X_domain,(1,N_domain)),(-1,d))
    XdXd1 = onp.tile(X_domain,(N_domain,1))
    
    XbXd0 = onp.reshape(onp.tile(X_boundary,(1,N_domain)),(-1,d))
    XbXd1 = onp.tile(X_domain,(N_boundary,1))
    
    XbXb0 = onp.reshape(onp.tile(X_boundary,(1,N_boundary)),(-1,d))
    XbXb1 = onp.tile(X_boundary,(N_boundary,1))
    
    arr_wx0 = onp.reshape(onp.tile(w0,(1,N_domain)),(-1,1))
    arr_wx1 = onp.reshape(onp.tile(w1,(1,N_domain)),(-1,1))
    arr_wxg = onp.reshape(onp.tile(wg,(1,N_domain)),(-1,d))
    arr_wy0 = onp.tile(w0,(N_domain,1))
    arr_wy1 = onp.tile(w1,(N_domain,1))
    arr_wyg = onp.tile(wg,(N_domain,1))
    
    arr_wy0_bd = onp.tile(w0,(N_boundary,1))
    arr_wy1_bd = onp.tile(w1,(N_boundary,1))
    arr_wyg_bd = onp.tile(wg,(N_boundary,1))
    
    val = vmap(lambda x,y,wx0,wx1,wxg,wy0,wy1,wyg: get_GNkernel_train(x,y,wx0,wx1,wxg,wy0,wy1,wyg,d,sigma))(XdXd0,XdXd1,arr_wx0,arr_wx1,arr_wxg,arr_wy0,arr_wy1,arr_wyg)
    Theta[:N_domain,:N_domain] = onp.reshape(val, (N_domain,N_domain))
    
    val = vmap(lambda x,y,wy0,wy1,wyg: get_GNkernel_train_boundary(x,y,wy0,wy1,wyg,d,sigma))(XbXd0,XbXd1,arr_wy0_bd,arr_wy1_bd,arr_wyg_bd)
    Theta[N_domain:,:N_domain] = onp.reshape(val, (N_boundary,N_domain))
    Theta[:N_domain,N_domain:] = onp.transpose(onp.reshape(val, (N_boundary,N_domain)))
    
    val = vmap(lambda x,y: kappa(x,y,d,sigma))(XbXb0, XbXb1)
    Theta[N_domain:,N_domain:] = onp.reshape(val, (N_boundary, N_boundary))
    return Theta
    
def assembly_Theta_value_predict(X_infer, X_domain, X_boundary, w0, w1, wg, sigma):
    N_infer, d = onp.shape(X_infer)
    N_domain, _ = onp.shape(X_domain)
    N_boundary, _ = onp.shape(X_boundary)
    Theta = onp.zeros((2*N_infer,N_domain+N_boundary))
    
    XiXd0 = onp.reshape(onp.tile(X_infer,(1,N_domain)),(-1,d))
    XiXd1 = onp.tile(X_domain,(N_infer,1))
    
    XiXb0 = onp.reshape(onp.tile(X_infer,(1,N_boundary)),(-1,d))
    XiXb1 = onp.tile(X_boundary,(N_infer,1))
    
    arr_wy0 = onp.tile(w0,(N_infer,1))
    arr_wy1 = onp.tile(w1,(N_infer,1))
    arr_wyg = onp.tile(wg,(N_infer,1))
    
    val = vmap(lambda x,y,wy0,wy1,wyg: get_GNkernel_val_predict(x,y,wy0,wy1,wyg,d,sigma))(XiXd0,XiXd1,arr_wy0,arr_wy1,arr_wyg)
    Theta[:N_infer,:N_domain] = onp.reshape(val, (N_infer,N_domain))
    
    val = vmap(lambda x,y: kappa(x,y,d,sigma))(XiXb0, XiXb1)
    Theta[:N_infer,N_domain:] = onp.reshape(val, (N_infer,N_boundary))
    
    val = vmap(lambda x,y,wy0,wy1,wyg: get_GNkernel_val_predict_Delta(x,y,wy0,wy1,wyg,d,sigma))(XiXd0,XiXd1,arr_wy0,arr_wy1,arr_wyg)
    Theta[N_infer:,:N_domain] = onp.reshape(val, (N_infer,N_domain))
    
    val = vmap(lambda x,y: Delta_x_kappa(x,y,d,sigma))(XiXb0, XiXb1)
    Theta[N_infer:,N_domain:] = onp.reshape(val, (N_infer,N_boundary))
    
    
    return Theta

def GPsolver(X_domain, X_boundary, X_test, sigma, nugget, sol_init, GN_step = 4):
    N_domain, d = onp.shape(X_domain)
    sol = sol_init
    rhs_f = vmap(f)(X_domain)[:,onp.newaxis]
    bdy_g = vmap(g)(X_boundary)[:,onp.newaxis]
    wg = -vmap(grad_a)(X_domain) #size?
    w1 = -vmap(a)(X_domain)[:,onp.newaxis]
    time_begin = time()
    for i in range(GN_step):
        
        w0 = alpha*m*(sol**(m-1))
        Theta_train = assembly_Theta(X_domain, X_boundary, w0, w1, wg, sigma)
        Theta_test = assembly_Theta_value_predict(X_domain, X_domain, X_boundary, w0, w1, wg, sigma)
        rhs = rhs_f + alpha*(m-1)*(sol**m)
        rhs = onp.concatenate((rhs, bdy_g), axis = 0)
        sol = Theta_test[:N_domain,:] @ (onp.linalg.solve(Theta_train + nugget*onp.diag(onp.diag(Theta_train)),rhs))
        total_mins = (time() - time_begin) / 60
        logging.info(f'[Timer] GP iteration {i+1}/{GN_step}, finished in {total_mins:.2f} minutes')
    Theta_test = assembly_Theta_value_predict(X_test, X_domain, X_boundary, w0, w1, wg, sigma)
    result_test = Theta_test @ (onp.linalg.solve(Theta_train + nugget*onp.diag(onp.diag(Theta_train)),rhs))
    N_infer, d = onp.shape(X_test)
    sol_test = result_test[:N_infer]
    Delta_sol_test = result_test[N_infer:]
    return sol, sol_test, Delta_sol_test

# def sample_points(N_domain, N_boundary, d, choice = 'random'):
#     X_domain = onp.zeros((N_domain,d))
#     X_boundary = onp.zeros((N_boundary,d))
    
#     X_domain = onp.random.randn(N_domain,d)  # N_domain*d
#     X_domain /= onp.linalg.norm(X_domain, axis=1)[:,onp.newaxis] # the divisor is of N_domain*1
#     random_radii = onp.random.rand(N_domain,1) ** (1/d)
#     X_domain *= random_radii
    
#     X_boundary = onp.random.randn(N_boundary,d)
#     X_boundary /= onp.linalg.norm(X_boundary, axis=1)[:,onp.newaxis]
#     return X_domain, X_boundary


def sample_points(N_domain, N_boundary, d, choice = 'random'):
    x1l = 0.0
    x1r = 1.0
    x2l = 0.0
    x2r = 1.0
    #(x,y) in [x1l,x1r]*[x2l,x2r] default = [0,1]*[0,1]
    # interior nodes
    X_domain = onp.concatenate((random.uniform(x1l, x1r, (N_domain, 1)), random.uniform(x2l, x2r, (N_domain, 1))), axis = 1)
    
    N_boundary_per_bd = int(N_boundary/4)
    X_boundary = onp.zeros((N_boundary_per_bd*4, 2))
    
    # bottom face
    X_boundary[0:N_boundary_per_bd, 0] = random.uniform(x1l, x1r, N_boundary_per_bd)
    X_boundary[0:N_boundary_per_bd, 1] = x2l
    # right face
    X_boundary[N_boundary_per_bd:2*N_boundary_per_bd, 0] = x1r
    X_boundary[N_boundary_per_bd:2*N_boundary_per_bd, 1] = random.uniform(x2l, x2r, N_boundary_per_bd)
    # top face
    X_boundary[2*N_boundary_per_bd:3*N_boundary_per_bd, 0] = random.uniform(x1l, x1r, N_boundary_per_bd)
    X_boundary[2*N_boundary_per_bd:3*N_boundary_per_bd, 1] = x2r
    # left face
    X_boundary[3*N_boundary_per_bd:4*N_boundary_per_bd, 1] = random.uniform(x2l, x2r, N_boundary_per_bd)
    X_boundary[3*N_boundary_per_bd:4*N_boundary_per_bd, 0] = x1l
        
    return X_domain, X_boundary


def logger(args, level = 'INFO'):
    log_root = args.logroot + 'NonVarLinElliptic_rate'
    log_name = 'dim' + str(args.dim) + '_kernel' + str(args.kernel)
    logdir = os.path.join(log_root, log_name)
    os.makedirs(logdir, exist_ok=True)
    log_para = 's' + str(args.sigma_scale) + str(args.nugget).replace(".","") + '_fa' + str(args.freq_a) + '_cos' + '_nexp' + str(args.num_exp)
    date = str(datetime.datetime.now())
    log_base = date[date.find("-"):date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
    filename = log_para + '_' + log_base + '.log'
    logging.basicConfig(level=logging.__dict__[level],
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(logdir+'/'+filename),
        logging.StreamHandler()]
    )
    return logdir+'/'+filename

def set_random_seeds(args):
    random_seed = args.randomseed
    random.seed(random_seed)

## get argument parser
args = get_parser()
filename = logger(args, level = 'INFO')
logging.info(f'argument is {args}')

def a(x):
    # return jnp.exp(jnp.sin(jnp.sum(args.freq_a * jnp.cos(x))))
    return 1.0
def grad_a(x):
    return grad(a)(x)

freq = 600
arr_s = [4,5,6,7,8,9]
# arr_s = [4]

num_s = onp.size(arr_s)
def u(x):

    ans = 0
    for k in range(1,freq):
        ans += jnp.prod(jnp.sin(jnp.pi*k*x))/(k**s) 
        # H^t norm squared is sum 1/k^{2s-2t}, so in H^{s-1/2}
    return 1/(2*jnp.pi**2)*ans

def Delta_u(x):
    return jnp.trace(hessian(u)(x))


def f(x):
    return -a(x) * jnp.trace(hessian(u)(x))- jnp.sum(grad(a)(x) * grad(u)(x)) + alpha*(u(x)**m)
    
def g(x):
    return u(x)

alpha = args.alpha
m = args.m

logging.info(f"[Equation] alpha: {alpha}, m: {m}")
logging.info(f"[Function] frequency of a: {args.freq_a}, regularity of u {arr_s}")

if args.kernel == "gaussian":
    from kernels.Gaussian_kernel import *
elif args.kernel == "inv_quadratics":
    from kernels.inv_quadratics import *
elif args.kernel == "Matern_3half":
    from kernels.Matern_3half import *
elif args.kernel == "Matern_5half":
    from kernels.Matern_5half import *
elif args.kernel == "Matern_7half":
    from kernels.Matern_7half import *
elif args.kernel == "Matern_9half":
    from kernels.Matern_9half import *
elif args.kernel == "Matern_11half":
    from kernels.Matern_11half import *


arr_N_domain = [1000,2000,4000,8000]
arr_N_boundary = [200, 400, 800, 1600]
# arr_N_domain = [1000,2000]
# arr_N_boundary = [200,400]
num_N = onp.size(arr_N_domain)
N_test = 8000
# ratio = args.sigma_scale

nugget = args.nugget
GN_step = args.GNsteps

logging.info(f"***** Total number of random experiments {args.num_exp} *****")

train_err_2_all = onp.zeros((num_N,num_s,args.num_exp))
train_err_inf_all = onp.zeros((num_N,num_s,args.num_exp))

test_err_2_all = onp.zeros((num_N,num_s,args.num_exp))
test_err_inf_all = onp.zeros((num_N,num_s,args.num_exp))
test_err_2_Delta_all = onp.zeros((num_N,num_s,args.num_exp))
test_err_inf_Delta_all = onp.zeros((num_N,num_s,args.num_exp))

d = args.dim
# sigma = ratio*onp.sqrt(d)
sigma = 0.2

logging.info(f'GN step: {GN_step}, d: {d}, sigma: {sigma}, number of points: N_domain {arr_N_domain}, N_boundary {arr_N_boundary}, N_test {N_test}, kernel: {args.kernel}, nugget: {args.nugget}')

for idx_exp in range(args.num_exp):
    logging.info(f"[Experiment] number: {idx_exp}")
    args.randomseed = idx_exp
    set_random_seeds(args)
    logging.info(f"[Seeds] random seeds: {args.randomseed}")
    
    for idx_N in range(num_N):
        N_domain = arr_N_domain[idx_N]
        N_boundary = arr_N_boundary[idx_N]
        X_domain, X_boundary = sample_points(N_domain, N_boundary, d, choice = 'random')
        X_test, _ = sample_points(N_test, N_boundary, d, choice = 'random')
    
        for idx_s in range(num_s):
            s = arr_s[idx_s]
            sol_init = onp.zeros((N_domain,1))
            
            sol, sol_test, Delta_sol_test = GPsolver(X_domain, X_boundary, X_test, sigma, nugget, sol_init, GN_step = GN_step)
            # print(Delta_sol_test)
            logging.info('[Calculating errs at collocation points ...]')
            
            # train
            sol_truth = vmap(u)(X_domain)[:,onp.newaxis]
            err = abs(sol-sol_truth)
            err_2 = onp.linalg.norm(err,'fro')/onp.sqrt(N_domain)
            train_err_2_all[idx_N,idx_s,idx_exp] = err_2
            err_inf = onp.max(err)
            train_err_inf_all[idx_N,idx_s,idx_exp] = err_inf
            
            
            # test
            sol_truth = vmap(u)(X_test)[:,onp.newaxis]
            err = abs(sol_test-sol_truth)
            err_2 = onp.linalg.norm(err,'fro')/onp.sqrt(N_test)
            test_err_2_all[idx_N,idx_s,idx_exp] = err_2
            err_inf = onp.max(err)
            test_err_inf_all[idx_N,idx_s,idx_exp] = err_inf
            
            Delta_u_truth = vmap(Delta_u)(X_test)[:,onp.newaxis]
            err = abs(Delta_sol_test-Delta_u_truth)
            err_2 = onp.linalg.norm(err,'fro')/onp.sqrt(N_test)
            test_err_2_Delta_all[idx_N,idx_s,idx_exp] = err_2
            err_inf = onp.max(err)
            test_err_inf_Delta_all[idx_N,idx_s,idx_exp] = err_inf
            
            logging.info(f'[L infinity error] train {train_err_inf_all[idx_N,idx_s,idx_exp]}, test {test_err_inf_all[idx_N,idx_s,idx_exp]}')
            logging.info(f'[L2 error] train {train_err_2_all[idx_N,idx_s,idx_exp]}, test {test_err_2_all[idx_N,idx_s,idx_exp]}')
            logging.info(f'[W2 error] test H2 {test_err_2_Delta_all[idx_N,idx_s,idx_exp]}, test W2inf{test_err_inf_Delta_all[idx_N,idx_s,idx_exp]}')
            
    
onp.savez(filename, trainLinf = train_err_inf_all, trainL2 = train_err_2_all, testLinf = test_err_inf_all, testL2 = test_err_2_all, testH2 = test_err_2_Delta_all, testW2inf = test_err_inf_Delta_all)