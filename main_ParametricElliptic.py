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

# solving -grad(a*grad u) + alpha u^m = f on unit ball
# a = a(x,theta)
def get_parser():
    parser = argparse.ArgumentParser(description='Parametric Elliptic equation GP solver')
    parser.add_argument("--alpha", type=float, default = 1.0)
    parser.add_argument("--m", type = int, default = 3)
    parser.add_argument("--dim_x", type = int, default = 2)
    parser.add_argument("--dim_theta", type = int, default = 1)
    parser.add_argument("--kernel", type=str, default="anisotripic_Gaussian")
    
    parser.add_argument("--sigma-scale_x", type = float, default = 0.25)
    parser.add_argument("--sigma-scale_theta", type = float, default = 0.1)
    # sigma_x = args.sigma-scale_x*sqrt(dim)
    # sigma_theta = args.sigma-scale_theta*sqrt(dim)
    
    parser.add_argument("--N_domain", type = int, default = 2000)
    parser.add_argument("--N_boundary", type = int, default = 400)
    parser.add_argument("--nugget", type = float, default = 1e-10)
    parser.add_argument("--GNsteps", type = int, default = 4)
    parser.add_argument("--logroot", type=str, default='./logs/')
    parser.add_argument("--randomseed", type=int, default=1)
    parser.add_argument("--num_exp", type=int, default=4)
    args = parser.parse_args()    
    return args

@jit # tx is short hand of theta_x
def get_GNkernel_train(x,tx,y,ty,wx0,wx1,wxg,wy0,wy1,wyg,sigma):
    # wx0 * delta_x + wxg * nabla_x delta_x + wx1 * Delta_x delta_x 
    return wx0*wy0*kappa(x,tx,y,ty,sigma) + wx0*wy1*Delta_y1_kappa(x,tx,y,ty,sigma) + wy0*wx1*Delta_x1_kappa(x,tx,y,ty,sigma) + wx1*wy1*Delta_x1_Delta_y1_kappa(x,tx,y,ty,sigma) + wx0*D_wy1_kappa(x,tx,y,ty,sigma,wyg) + wy0*D_wx1_kappa(x,tx,y,ty,sigma,wxg) + wx1*Delta_x1_D_wy1_kappa(x,tx,y,ty,sigma,wyg) + wy1*D_wx1_Delta_y1_kappa(x,tx,y,ty,sigma,wxg) + D_wx1_D_wy1_kappa(x,tx,y,ty,sigma,wxg,wyg)

@jit
def get_GNkernel_train_boundary(x,tx,y,ty,wy0,wy1,wyg,sigma):
    return wy0*kappa(x,tx,y,ty,sigma) + wy1*Delta_y1_kappa(x,tx,y,ty,sigma) + D_wy1_kappa(x,tx,y,ty,sigma,wyg)
@jit
def get_GNkernel_val_predict(x,tx,y,ty,wy0,wy1,wyg,sigma):
    return wy0*kappa(x,tx,y,ty,sigma) + wy1*Delta_y1_kappa(x,tx,y,ty,sigma) + D_wy1_kappa(x,tx,y,ty,sigma,wyg)


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
    arr_wxg = onp.reshape(onp.tile(wg,(1,N_domain)),(-1,d_x))
    arr_wy0 = onp.tile(w0,(N_domain,1))
    arr_wy1 = onp.tile(w1,(N_domain,1))
    arr_wyg = onp.tile(wg,(N_domain,1))
    
    arr_wy0_bd = onp.tile(w0,(N_boundary,1))
    arr_wy1_bd = onp.tile(w1,(N_boundary,1))
    arr_wyg_bd = onp.tile(wg,(N_boundary,1))
    
    val = vmap(lambda x,tx,y,ty,wx0,wx1,wxg,wy0,wy1,wyg: get_GNkernel_train(x,tx,y,ty,wx0,wx1,wxg,wy0,wy1,wyg,sigma))(XdXd0[:,:d_x],XdXd0[:,d_x:],XdXd1[:,:d_x],XdXd1[:,d_x:],arr_wx0,arr_wx1,arr_wxg,arr_wy0,arr_wy1,arr_wyg)
    Theta[:N_domain,:N_domain] = onp.reshape(val, (N_domain,N_domain))
    
    val = vmap(lambda x,tx,y,ty,wy0,wy1,wyg: get_GNkernel_train_boundary(x,tx,y,ty,wy0,wy1,wyg,sigma))(XbXd0[:,:d_x],XbXd0[:,d_x:],XbXd1[:,:d_x],XbXd1[:,d_x:],arr_wy0_bd,arr_wy1_bd,arr_wyg_bd)
    Theta[N_domain:,:N_domain] = onp.reshape(val, (N_boundary,N_domain))
    Theta[:N_domain,N_domain:] = onp.transpose(onp.reshape(val, (N_boundary,N_domain)))
    
    val = vmap(lambda x1,x2,y1,y2: kappa(x1, x2, y1, y2, sigma))(XbXb0[:,:d_x], XbXb0[:,d_x:], XbXb1[:,:d_x], XbXb1[:,d_x:])
    Theta[N_domain:,N_domain:] = onp.reshape(val, (N_boundary, N_boundary))
    return Theta

def assembly_Theta_value_predict(X_infer, X_domain, X_boundary, w0, w1, wg, sigma):
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
    arr_wyg = onp.tile(wg,(N_infer,1))
    
    val = vmap(lambda x,tx,y,ty,wy0,wy1,wyg: get_GNkernel_val_predict(x,tx,y,ty,wy0,wy1,wyg,sigma))(XiXd0[:,:d_x],XiXd0[:,d_x:],XiXd1[:,:d_x],XiXd1[:,d_x:],arr_wy0,arr_wy1,arr_wyg)
    Theta[:N_infer,:N_domain] = onp.reshape(val, (N_infer,N_domain))
    
    val = vmap(lambda x,tx,y,ty: kappa(x,tx,y,ty,sigma))(XiXb0[:,:d_x], XiXb0[:,d_x:],XiXb1[:,:d_x],XiXb1[:,d_x:])
    Theta[:N_infer,N_domain:] = onp.reshape(val, (N_infer,N_boundary))
    return Theta

def GPsolver(X_domain, X_boundary, sigma, nugget, sol_init, GN_step = 4):
    # N_domain, d = onp.shape(X_domain)
    sol = sol_init
    rhs_f = vmap(f)(X_domain[:,:d_x],X_domain[:,d_x:])[:,onp.newaxis]
    bdy_g = vmap(g)(X_boundary[:,:d_x],X_boundary[:,d_x:])[:,onp.newaxis]
    wg = vmap(gradx_a)(X_domain[:,:d_x],X_domain[:,d_x:]) #size?

    w1 = -vmap(a)(X_domain[:,:d_x],X_domain[:,d_x:])[:,onp.newaxis]
    time_begin = time()
    for i in range(GN_step):
        
        w0 = alpha*m*(sol**(m-1))
        Theta_train = assembly_Theta(X_domain, X_boundary, w0, w1, wg, sigma)
        Theta_test = assembly_Theta_value_predict(X_domain, X_domain, X_boundary, w0, w1, wg, sigma)
        rhs = rhs_f + alpha*(m-1)*(sol**m)
        rhs = onp.concatenate((rhs, bdy_g), axis = 0)
        sol = Theta_test @ (onp.linalg.solve(Theta_train + nugget*onp.diag(onp.diag(Theta_train)),rhs))
        total_mins = (time() - time_begin) / 60
        logging.info(f'[Timer] GP iteration {i+1}/{GN_step}, finished in {total_mins:.2f} minutes')
    return sol

def sample_points(N_domain, N_boundary, d_x, d_theta, choice = 'random'):
    X_domain = onp.zeros((N_domain,d_x+d_theta))
    X_boundary = onp.zeros((N_boundary,d_x+d_theta))
    
    X_domain[:,:d_x] = onp.random.randn(N_domain,d_x)  # N_domain*d
    X_domain[:,:d_x] /= onp.linalg.norm(X_domain[:,:d_x], axis=1)[:,onp.newaxis] # the divisor is of N_domain*1
    random_radii = onp.random.rand(N_domain,1) ** (1/d_x)
    X_domain[:,:d_x] *= random_radii
    
    X_domain[:,d_x:] = onp.random.randn(N_domain,d_theta)  # N_domain*d
    X_domain[:,:d_x:] /= onp.linalg.norm(X_domain[:,d_x:], axis=1)[:,onp.newaxis] # the divisor is of N_domain*1
    random_radii = onp.random.rand(N_domain,1) ** (1/d_theta)
    X_domain[:,d_x:] *= random_radii
    
    X_boundary[:,:d_x] = onp.random.randn(N_boundary,d_x)
    X_boundary[:,:d_x] /= onp.linalg.norm(X_boundary[:,:d_x], axis=1)[:,onp.newaxis]
    X_boundary[:,d_x:] = onp.random.randn(N_boundary,d_theta)
    X_boundary[:,:d_x:] /= onp.linalg.norm(X_boundary[:,d_x:], axis=1)[:,onp.newaxis] # the divisor is of N_domain*1
    random_radii = onp.random.rand(N_boundary,1) ** (1/d_theta)
    X_boundary[:,d_x:] *= random_radii
    
    return X_domain, X_boundary

def logger(args, level = 'INFO'):
    log_root = args.logroot + 'ParametricElliptic'
    log_name = 'dim_x' + str(args.dim_x) + 'dim_theta' + str(args.dim_theta) + '_kernel' + str(args.kernel)
    logdir = os.path.join(log_root, log_name)
    os.makedirs(logdir, exist_ok=True)
    log_para = 'alpha' + str(args.alpha) + 'm' + str(args.m) + 'sigma-scale' + str(args.sigma_scale_x) +'_' +str(args.sigma_scale_theta) + '_Ndomain' + str(args.N_domain) + '_Nbd' + str(args.N_boundary) + '_nugget' + str(args.nugget).replace(".","") + '_numexp' + str(args.num_exp)
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
    
    d_x = args.dim_x
    d_theta = args.dim_theta
    alpha = args.alpha
    m = args.m
    
    @jit
    def a(x,theta):
        return jnp.exp(jnp.sum(jnp.sin(x))) * jnp.exp(jnp.sum(jnp.cos(theta)))
    @jit
    def gradx_a(x,theta):
        return grad(a,0)(x,theta)
    @jit
    def u(x,theta):
        return jnp.sum(jnp.sin(x)) * jnp.sum(jnp.sin(theta))
    @jit
    def gradx_u(x,theta):
        return grad(u,0)(x,theta)
    @jit
    def f(x,theta):
        return -a(x,theta) * jnp.trace(hessian(lambda x: u(x,theta))(x))+ jnp.sum(gradx_a(x,theta) * gradx_u(x,theta)) + alpha*(u(x,theta)**m)
    @jit
    def g(x,theta):
        return u(x,theta)
    
    
    logging.info(f"[Equation] alpha: {alpha}, m: {m}")
    
    if args.kernel == "anisotripic_Gaussian":
        from kernels.anisotripic_Gaussian_kernel import *

    
    d_x = args.dim_x
    d_theta = args.dim_theta
    N_domain = args.N_domain
    N_boundary = args.N_boundary
    sigma = [args.sigma_scale_x*onp.sqrt(d_x),args.sigma_scale_theta*onp.sqrt(d_theta)]
    nugget = args.nugget
    GN_step = args.GNsteps

    logging.info(f'GN step: {GN_step}, dx: {d_x}, d_theta:{d_theta} sigma: {sigma}, number of points: N_domain {N_domain}, N_boundary {N_boundary}, kernel: {args.kernel}, nugget: {args.nugget}')
    
    
    logging.info(f"***** Total number of random experiments {args.num_exp} *****")
    
    err_2_all = []
    err_inf_all = []
    for idx_exp in range(args.num_exp):
        logging.info(f"[Experiment] number: {idx_exp}")
        args.randomseed = idx_exp
        set_random_seeds(args)
        logging.info(f"[Seeds] random seeds: {args.randomseed}")
        
        X_domain, X_boundary = sample_points(N_domain, N_boundary, d_x, d_theta, choice = 'random')
        
        sol_init = onp.random.randn(args.N_domain,1)
        
        sol = GPsolver(X_domain, X_boundary, sigma, nugget, sol_init, GN_step = GN_step)

        logging.info('[Calculating errs at collocation points ...]')
        sol_truth = vmap(u)(X_domain[:,:d_x],X_domain[:,d_x:])[:,onp.newaxis]
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
    
    