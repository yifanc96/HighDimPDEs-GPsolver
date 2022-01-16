import jax.numpy as jnp
from jax import grad, vmap, hessian, jit

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

# solving -grad(a*grad u) + alpha u^m = f on torus, to be completed
def get_parser():
    parser = argparse.ArgumentParser(description='NonLinElliptic equation GP solver')
    parser.add_argument("--freq_a", type=float, default = 1.0)
    parser.add_argument("--freq_u", type=float, default = 4.0)
    parser.add_argument("--alpha", type=float, default = 1.0)
    parser.add_argument("--m", type = int, default = 3)
    parser.add_argument("--dim", type = int, default = 1)
    parser.add_argument("--kernel", type=str, default="periodic")
    parser.add_argument("--sigma-scale", type = float, default = 0.25)
    # sigma = args.sigma-scale*sqrt(dim)
    
    parser.add_argument("--N_domain", type = int, default = 1000)
    parser.add_argument("--nugget", type = float, default = 1e-10)
    parser.add_argument("--GNsteps", type = int, default = 4)
    parser.add_argument("--logroot", type=str, default='./logs/')
    parser.add_argument("--randomseed", type=int, default=9999)
    parser.add_argument("--num_exp", type=int, default=2)
    args = parser.parse_args()    
    return args
@jit
def get_GNkernel_train(x,y,wx0,wx1,wxg,wy0,wy1,wyg,d,sigma):
    # wx0 * delta_x + wxg * nabla delta_x + wx1 * Delta delta_x 
    return wx0*wy0*kappa(x,y,d,sigma) + wx0*wy1*Delta_y_kappa(x,y,d,sigma) + wy0*wx1*Delta_x_kappa(x,y,d,sigma) + wx1*wy1*Delta_x_Delta_y_kappa(x,y,d,sigma) + wx0*D_wy_kappa(x,y,d,sigma,wyg) + wy0*D_wx_kappa(x,y,d,sigma,wxg) + wx1*Delta_x_D_wy_kappa(x,y,d,sigma,wyg) + wy1*D_wx_Delta_y_kappa(x,y,d,sigma,wxg) + D_wx_D_wy_kappa(x,y,d,sigma,wxg,wyg)
@jit
def get_GNkernel_train_boundary(x,y,wy0,wy1,wyg,d,sigma):
    return wy0*kappa(x,y,d,sigma) + wy1*Delta_y_kappa(x,y,d,sigma) + D_wy_kappa(x,y,d,sigma,wyg)
@jit
def get_GNkernel_val_predict(x,y,wy0,wy1,wyg,d,sigma):
    return wy0*kappa(x,y,d,sigma) + wy1*Delta_y_kappa(x,y,d,sigma) + D_wy_kappa(x,y,d,sigma,wyg)


def assembly_Theta(X_domain, w0, w1, wg, sigma):
    # X_domain, dim: N_domain*d; 
    # w0 col vec: coefs of Diracs, dim: N_domain; 
    # w1 coefs of Laplacians, dim: N_domain
    
    N_domain,d = onp.shape(X_domain)
    Theta = onp.zeros((N_domain,N_domain))
    
    XdXd0 = onp.reshape(onp.tile(X_domain,(1,N_domain)),(-1,d))
    XdXd1 = onp.tile(X_domain,(N_domain,1))
    
    arr_wx0 = onp.reshape(onp.tile(w0,(1,N_domain)),(-1,1))
    arr_wx1 = onp.reshape(onp.tile(w1,(1,N_domain)),(-1,1))
    arr_wxg = onp.reshape(onp.tile(wg,(1,N_domain)),(-1,d))
    arr_wy0 = onp.tile(w0,(N_domain,1))
    arr_wy1 = onp.tile(w1,(N_domain,1))
    arr_wyg = onp.tile(wg,(N_domain,1))
    
    val = vmap(lambda x,y,wx0,wx1,wxg,wy0,wy1,wyg: get_GNkernel_train(x,y,wx0,wx1,wxg,wy0,wy1,wyg,d,sigma))(XdXd0,XdXd1,arr_wx0,arr_wx1,arr_wxg,arr_wy0,arr_wy1,arr_wyg)
    Theta[:N_domain,:N_domain] = onp.reshape(val, (N_domain,N_domain))
    
    return Theta
    
def assembly_Theta_value_predict(X_infer, X_domain, w0, w1, wg, sigma):
    N_infer, d = onp.shape(X_infer)
    N_domain, _ = onp.shape(X_domain)
    Theta = onp.zeros((N_infer,N_domain))
    
    XiXd0 = onp.reshape(onp.tile(X_infer,(1,N_domain)),(-1,d))
    XiXd1 = onp.tile(X_domain,(N_infer,1))
    
    
    arr_wy0 = onp.tile(w0,(N_infer,1))
    arr_wy1 = onp.tile(w1,(N_infer,1))
    arr_wyg = onp.tile(wg,(N_infer,1))
    
    val = vmap(lambda x,y,wy0,wy1,wyg: get_GNkernel_val_predict(x,y,wy0,wy1,wyg,d,sigma))(XiXd0,XiXd1,arr_wy0,arr_wy1,arr_wyg)
    Theta[:N_infer,:N_domain] = onp.reshape(val, (N_infer,N_domain))
    
    return Theta

def GPsolver(X_domain, sigma, nugget, sol_init, GN_step = 4):
    # N_domain, d = onp.shape(X_domain)
    sol = sol_init
    rhs_f = vmap(f)(X_domain)[:,onp.newaxis]
    wg = vmap(grad_a)(X_domain) #size?
    w1 = -vmap(a)(X_domain)[:,onp.newaxis]
    time_begin = time()
    for i in range(GN_step):
        
        w0 = alpha*m*(sol**(m-1))
        Theta_train = assembly_Theta(X_domain, w0, w1, wg, sigma)
        Theta_test = assembly_Theta_value_predict(X_domain, X_domain, w0, w1, wg, sigma)
        rhs = rhs_f + alpha*(m-1)*(sol**m)
        sol = Theta_test @ (onp.linalg.solve(Theta_train + nugget*onp.diag(onp.diag(Theta_train)),rhs))
        total_mins = (time() - time_begin) / 60
        logging.info(f'[Timer] GP iteration {i+1}/{GN_step}, finished in {total_mins:.2f} minutes')
    return sol

def sample_points(N_domain, d, choice = 'random'):
    X_domain = onp.random.uniform(low=0.0, high=1.0, size=(N_domain,d))
    return X_domain

def logger(args, level = 'INFO'):
    log_root = args.logroot + 'VarCoefEllipticTorus'
    log_name = 'dim' + str(args.dim) + '_kernel' + str(args.kernel)
    logdir = os.path.join(log_root, log_name)
    os.makedirs(logdir, exist_ok=True)
    log_para = 'alpha' + str(args.alpha) + 'm' + str(args.m) + 'sigma-scale' + str(args.sigma_scale) + '_Ndomain' + str(args.N_domain) + '_nugget' + str(args.nugget).replace(".","") + '_freqa' + str(args.freq_a) + '_frequ' + str(args.freq_u) + '_numexp' + str(args.num_exp)
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

if __name__ == '__main__':
    ## get argument parser
    args = get_parser()
    filename = logger(args, level = 'INFO')
    logging.info(f'argument is {args}')
    
    @jit
    def a(x):
        return jnp.exp(jnp.sin(jnp.sum(args.freq_a * jnp.cos(2*jnp.pi*x))))
    @jit
    def grad_a(x):
        return grad(a)(x)
    @jit
    def u(x):
        return jnp.sin(jnp.sum(args.freq_u * jnp.cos(2*jnp.pi*x)))
    @jit
    def f(x):
        return -a(x) * jnp.trace(hessian(u)(x))+ jnp.sum(grad(a)(x) * grad(u)(x)) + alpha*(u(x)**m)
    @jit
    def g(x):
        return u(x)
    
    alpha = args.alpha
    m = args.m
    
    logging.info(f"[Equation] alpha: {alpha}, m: {m}")
    logging.info(f"[Function] frequency of a: {args.freq_a}, frequency of u: {args.freq_u}")
    
    if args.kernel == "periodic":
        from kernels.periodic_kernel import *
    d = args.dim
    N_domain = args.N_domain
    ratio = args.sigma_scale
    sigma = ratio*onp.sqrt(d)
    nugget = args.nugget
    GN_step = args.GNsteps

    logging.info(f'GN step: {GN_step}, d: {d}, sigma: {sigma}, number of points: N_domain {N_domain}, kernel: {args.kernel}, nugget: {args.nugget}')
    
    
    logging.info(f"***** Total number of random experiments {args.num_exp} *****")
    
    err_2_all = []
    err_inf_all = []
    for idx_exp in range(args.num_exp):
        logging.info(f"[Experiment] number: {idx_exp}")
        args.randomseed = idx_exp
        set_random_seeds(args)
        logging.info(f"[Seeds] random seeds: {args.randomseed}")
        
        X_domain = sample_points(N_domain, d, choice = 'random')
        
        sol_init = onp.random.randn(args.N_domain,1)
        
        sol = GPsolver(X_domain, sigma, nugget, sol_init, GN_step = GN_step)

        logging.info('[Calculating errs at collocation points ...]')
        sol_truth = vmap(u)(X_domain)[:,onp.newaxis]
        err = abs(sol-sol_truth)
        err_2 = onp.linalg.norm(err,'fro')/(N_domain)
        err_2_all.append(err_2)
        err_inf = onp.max(err)
        err_inf_all.append(err_inf)
        logging.info(f'[L infinity error] {err_inf}')
        logging.info(f'[L2 error] {err_2}')
    
    logging.info(f'[Average L infinity error] {onp.mean(err_inf_all)}')
    logging.info(f'[Average L2 error] {onp.mean(err_2_all)}')
    
    onp.savez(filename, Linf = err_inf_all, L2 = err_2_all)