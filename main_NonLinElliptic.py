import jax.numpy as jnp
from jax import grad, vmap, hessian, jit

import jax.ops as jop
from jax.config import config; 
config.update("jax_enable_x64", True)

# numpy
import numpy as onp
from numpy import random 

import argparse
import logging
import datetime
from time import time
import os

# solving -Delta u + alpha u^m = f
def get_parser():
    parser = argparse.ArgumentParser(description='NonLinElliptic equation GP solver')
    parser.add_argument("--alpha", type=float, default = 1.0)
    parser.add_argument("--m", type = int, default = 3)
    parser.add_argument("--dim", type = int, default = 2)
    parser.add_argument("--kernel", type=str, default="Matern_5half", choices=["gaussian","inv_quadratics","Matern_3half","Matern_5half","Matern_7half","Matern_9half","Matern_11half"])
    parser.add_argument("--sigma-scale", type = float, default = 0.25)
    # sigma = args.sigma-scale*sqrt(dim)
    
    parser.add_argument("--N_domain", type = int, default = 2000)
    parser.add_argument("--N_boundary", type = int, default = 400)
    parser.add_argument("--nugget", type = float, default = 1e-10)
    parser.add_argument("--GNsteps", type = int, default = 4)
    parser.add_argument("--logroot", type=str, default='./logs/')
    parser.add_argument("--randomseed", type=int, default=9999)
    args = parser.parse_args()    
    return args

@jit
def get_GNkernel_train(x,y,wx0,wx1,wy0,wy1,d,sigma):
    return wx0*wy0*kappa(x,y,d,sigma) + wx0*wy1*Delta_y_kappa(x,y,d,sigma) + wy0*wx1*Delta_x_kappa(x,y,d,sigma) + wx1*wy1*Delta_x_Delta_y_kappa(x,y,d,sigma)
@jit
def get_GNkernel_train_boundary(x,y,wy0,wy1,d,sigma):
    return wy0*kappa(x,y,d,sigma) + wy1*Delta_y_kappa(x,y,d,sigma)
@jit
def get_GNkernel_val_predict(x,y,wy0,wy1,d,sigma):
    return wy0*kappa(x,y,d,sigma) + wy1*Delta_y_kappa(x,y,d,sigma)


def assembly_Theta(X_domain, X_boundary, w0, w1, sigma):
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
    arr_wy0 = onp.tile(w0,(N_domain,1))
    arr_wy1 = onp.tile(w1,(N_domain,1))
    
    arr_wy0_bd = onp.tile(w0,(N_boundary,1))
    arr_wy1_bd = onp.tile(w1,(N_boundary,1))
    
    val = vmap(lambda x,y,wx0,wx1,wy0,wy1: get_GNkernel_train(x,y,wx0,wx1,wy0,wy1,d,sigma))(XdXd0,XdXd1,arr_wx0,arr_wx1,arr_wy0,arr_wy1)
    Theta[:N_domain,:N_domain] = onp.reshape(val, (N_domain,N_domain))
    
    val = vmap(lambda x,y,wy0,wy1: get_GNkernel_train_boundary(x,y,wy0,wy1,d,sigma))(XbXd0,XbXd1,arr_wy0_bd,arr_wy1_bd)
    Theta[N_domain:,:N_domain] = onp.reshape(val, (N_boundary,N_domain))
    Theta[:N_domain,N_domain:] = onp.transpose(onp.reshape(val, (N_boundary,N_domain)))
    
    val = vmap(lambda x,y: kappa(x,y,d,sigma))(XbXb0, XbXb1)
    Theta[N_domain:,N_domain:] = onp.reshape(val, (N_boundary, N_boundary))
    return Theta
    
def assembly_Theta_value_predict(X_infer, X_domain, X_boundary, w0, w1, sigma):
    N_infer, d = onp.shape(X_infer)
    N_domain, _ = onp.shape(X_domain)
    N_boundary, _ = onp.shape(X_boundary)
    Theta = onp.zeros((N_infer,N_domain+N_boundary))
    
    XiXd0 = onp.reshape(onp.tile(X_infer,(1,N_domain)),(-1,d))
    XiXd1 = onp.tile(X_domain,(N_infer,1))
    
    XiXb0 = onp.reshape(onp.tile(X_infer,(1,N_boundary)),(-1,d))
    XiXb1 = onp.tile(X_boundary,(N_infer,1))
    
    arr_wy0 = onp.tile(w0,(N_infer,1))
    arr_wy1 = onp.tile(w1,(N_infer,1))
    
    val = vmap(lambda x,y,wy0,wy1: get_GNkernel_val_predict(x,y,wy0,wy1,d,sigma))(XiXd0,XiXd1,arr_wy0,arr_wy1)
    Theta[:N_infer,:N_domain] = onp.reshape(val, (N_infer,N_domain))
    
    val = vmap(lambda x,y: kappa(x,y,d,sigma))(XiXb0, XiXb1)
    Theta[:N_infer,N_domain:] = onp.reshape(val, (N_infer,N_boundary))
    return Theta

def GPsolver(X_domain, X_boundary, sigma, nugget, sol_init, GN_step = 4):
    N_domain, d = onp.shape(X_domain)
    sol = sol_init
    rhs_f = vmap(f)(X_domain)[:,onp.newaxis]
    bdy_g = vmap(g)(X_boundary)[:,onp.newaxis]
    time_begin = time()
    for i in range(GN_step):
        w1 = -onp.ones((N_domain,1))
        w0 = alpha*m*(sol**(m-1))
        Theta_train = assembly_Theta(X_domain, X_boundary, w0, w1, sigma)
        Theta_test = assembly_Theta_value_predict(X_domain, X_domain, X_boundary, w0, w1, sigma)
        rhs = rhs_f + alpha*(m-1)*(sol**m)
        rhs = onp.concatenate((rhs, bdy_g), axis = 0)
        sol = Theta_test @ (onp.linalg.solve(Theta_train + nugget*onp.diag(onp.diag(Theta_train)),rhs))
        total_mins = (time() - time_begin) / 60
        logging.info(f'[Timer] GP iteration {i+1}/{GN_step}, finished in {total_mins:.2f} minutes')
    return sol

def sample_points(N_domain, N_boundary, d, choice = 'random'):
    X_domain = onp.zeros((N_domain,d))
    X_boundary = onp.zeros((N_boundary,d))
    
    X_domain = onp.random.randn(N_domain,d)  # N_domain*d
    X_domain /= onp.linalg.norm(X_domain, axis=1)[:,onp.newaxis] # the divisor is of N_domain*1
    random_radii = onp.random.rand(N_domain,1) ** (1/d)
    X_domain *= random_radii
    
    X_boundary = onp.random.randn(N_boundary,d)
    X_boundary /= onp.linalg.norm(X_boundary, axis=1)[:,onp.newaxis]
    return X_domain, X_boundary

def logger(args, level = 'INFO'):
    log_root = args.logroot + 'NonLinElliptic'
    log_name = 'dim' + str(args.dim) + '_kernel' + str(args.kernel)
    logdir = os.path.join(log_root, log_name)
    os.makedirs(logdir, exist_ok=True)
    log_para = 'alpha' + str(args.alpha) + 'm' + str(args.m) + 'sigma-scale' + str(args.sigma_scale) + '_Ndomain' + str(args.N_domain) + '_Nbd' + str(args.N_boundary) + '_nugget' + str(args.nugget).replace(".","")
    date = str(datetime.datetime.now())
    log_base = date[date.find("-"):date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
    filename = log_para + '_' + log_base + '.log'
    logging.basicConfig(level=logging.__dict__[level],
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(logdir+'/'+filename),
        logging.StreamHandler()]
    )

def set_random_seeds(args):
    random_seed = args.randomseed
    random.seed(random_seed)
    
def u(x):
    return jnp.sin(jnp.sum(x))
def f(x):
    return -jnp.trace(hessian(u)(x))+alpha*(u(x)**m)
def g(x):
    return u(x)

if __name__ == '__main__':
    ## get argument parser
    args = get_parser()
    logger(args, level = 'INFO')
    logging.info(f'argument is {args}')
    
    alpha = args.alpha
    m = args.m
    
    logging.info(f"[Equation] alpha: {alpha}, m: {m}")
    set_random_seeds(args)
    logging.info(f"[Seeds] random seeds: {args.randomseed}")

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

    d = args.dim
    N_domain = args.N_domain
    N_boundary = args.N_boundary
    X_domain, X_boundary = sample_points(N_domain, N_boundary, d, choice = 'random')
    
    sol_init = onp.random.randn(args.N_domain,1)
    ratio = args.sigma_scale
    sigma = ratio*onp.sqrt(d)
    nugget = args.nugget
    GN_step = args.GNsteps

    logging.info(f'GN step: {GN_step}, d: {d}, sigma: {sigma}, number of points: N_domain {N_domain}, N_boundary {N_boundary}, kernel: {args.kernel}, nugget: {args.nugget}')
    
    sol = GPsolver(X_domain, X_boundary, sigma, nugget, sol_init, GN_step = GN_step)

    logging.info('[Calculating errs at collocation points ...]')
    sol_truth = vmap(u)(X_domain)[:,onp.newaxis]
    err = abs(sol-sol_truth)
    err_2 = onp.linalg.norm(err,'fro')/(N_domain)
    err_inf = onp.max(err)
    logging.info(f'[L infinity error] {err_inf}')
    logging.info(f'[L2 error] {err_2}')