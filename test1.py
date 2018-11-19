#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 18:55:03 2018

@author: jmw418
"""
###Packages###
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


### Initial condition as a function###
def jump(x, alpha=0.1, beta=0.3):
    ''' jump:
        
 SUMMARY: 
     This is a function that produces a square wave height 1 in the range between
     alpha and beta
     
 FLEXIBILITY:
     
     '''
    one = lambda x: 1
    return np.where((x<beta) & (x>=alpha), one(x), 0.)


###Timestepping methods###
def picard(X, nx, nt, Tfinal, mu, theta, form):
    """Picard:
        
SUMMARY:
    This is a timestepping algorithm that solves the burgers equation \
    when it is discretised by the picard method.

FLEXIBILITY:
    The 
    """
    ### Parameters ###
    dx = (1-0)/(nx-1) 
    dt = (Tfinal-0)/(nt-1)
    C = 1*dt/(2*dx)
    D = mu*dt/(dx**2)
    
    ### warnings ###
    ### Initialise structure ###
    beta = np.zeros([nx])
    A = np.zeros([nx,nx])
    if form == "non conservative":
        if theta >0: ###we need to invert a matrix 
            for i in range(0,nt-1):###time stepping###
        
                for j in range(0,nx): ### create vector on RHS###
                    beta[j] = X[j]\
                    - C*(1-theta)*X[j]*(X[(j+1)%nx] - X[(j-1)%nx])\
                    + (D)*(1-theta)*(X[(j-1)%nx]-2*X[j]+X[(j+1)%nx])
        
                for p in range(0,nx): ### Create Matrix on LHS ###
                    A[p,(p+1)%nx] =  theta*C*X[p] - theta*D # b 
                    A[p,(p-1)%nx] = -theta*C*X[p] - theta*D  #c  
                    A[p,p] = 1 + 2*theta*D
            
                X[:] = scipy.linalg.solve(A, beta)### Solving for next timestep ###
        if theta == 0: ## to avoid inverting a identity matrix pointlessly.
            for i in range(0,nt-1):###time stepping###
                for j in range(0,nx): ### create vector on RHS###
                    beta[j] = X[j]\
                    - C*(1-theta)*X[j]*(X[(j+1)%nx] - X[(j-1)%nx])\
                    + (D)*(1-theta)*(X[(j-1)%nx]-2*X[j]+X[(j+1)%nx])

                X[:] = beta ### Solving for next timestep ###
    ###
    if form == "conservative":
        if theta >0:
            for i in range(0,nt-1):###time stepping###
        
                for j in range(0,nx): ### create vector on RHS###
                    beta[j] = X[j]\
                    - C*0.5*(1)*(X[(j+1)%nx]**2 - X[(j-1)%nx]**2)\
                    + (D)*(1)*(X[(j-1)%nx]-2*X[j]+X[(j+1)%nx])
           
                for p in range(0,nx): ### Create Matrix on LHS ###
                    A[p,(p+1)%nx] =  0.5*theta*C*X[p] - theta*D # b 
                    A[p,(p-1)%nx] = -0.5*theta*C*X[p] - theta*D  #c  
                    A[p,p] = 1 + 2*theta*D + C*theta*0.5*(X[(p+1)%nx] - X[(p-1)%nx])
        
                X[:] = scipy.linalg.solve(A, beta)### Solving for next timestep ###
        ###
        if theta == 0: ## to avoid inverting a identity matrix pointlessly.
            for i in range(0,nt-1):###time stepping###
                for j in range(0,nx): ### create vector on RHS###
                    beta[j] = X[j]\
                    - C*0.5*(1)*(X[(j+1)%nx]**2 - X[(j-1)%nx]**2)\
                    + (D)*(1)*(X[(j-1)%nx]-2*X[j]+X[(j+1)%nx])
    
                X[:] = beta ### Solving for next timestep ###        
 
    return X

   
def newton(X, nx, nt, Tfinal, mu, theta, form):

    ### derived parameters ###
    dx = (1-0)/(nx-1) 
    dt = (Tfinal-0)/(nt-1)
    C = 1*dt/(2*dx)
    D = mu*dt/(dx**2)
    
    ##Creation of newton method structure 
    beta = np.zeros([nx])
    w = np.zeros([nx])
    dw = np.zeros([nx])
    if form == "non conservative":
        for i in range(0,nt-1):### time loop 
        
            for j in range(0,nx): 
                w[j] = X[j] ### first initialise w^0 = phi^n as initial guess###
                          ### this is the starting guess for newton method ###
        
            ### Construct the newton loop
            tol = 10**(-13)## extreeme accuracy can be imposed
            err  = 2*tol
            while (err>tol): ### until convergence ###
            
                ### Create beta^k_j ###
                for q in range(0,nx): 
                    beta[q] = X[q] - C*(1-theta)*X[q]*(X[(q+1)%nx] - X[(q-1)%nx]) \
                    + (D)*(1-theta)*(X[(q-1)%nx]-2*X[q]+X[(q+1)%nx]) \
                    -w[q] - C*theta*w[q]*(w[(q+1)%nx]-w[(q-1)%nx]) + D*theta*(w[(q+1)%nx]-2*w[q] +w[(q-1)%nx])
            
                ##Create A^k_j ###
                A = np.zeros([nx,nx])
                for p in range(1,nx):
                    A[p-1,p] = ((theta*C*w[p])-(theta*D)) # b
                for p in range(0,nx-1):
                    A[p+1,p] = ((-theta*C*w[p+1])-(theta*D)) # c
                for p in range(0,nx):
                    A[p,p] = (1+ 2*theta*D + C*theta*(w[(p+1)%nx]-w[(p-1)%nx]))
            
            
                ## solving for dw
                dw = scipy.linalg.solve(A, beta)
                err = np.linalg.norm(dw,2)
                ## improve the newton loop
                w = w + dw 
                ## we have w^k
            
    
        
            for j in range(0,nx): ##replace
                X[j] = w[j]
    if form == "conservative":
        for i in range(0,nt-1):### time loop 
            ## first initialise w^0 = phi^n
            for j in range(0,nx): 
                w[j] = X[j]

            ### Construct the newton loop
            tol = 10**(-13) ### accuracy can be imposed ###
            err  = 2*tol
            while (err>tol): ## eventually replace with while loop and stopping criterion

                ### Create beta^k ###
                for q in range(0,nx): 
                    beta[q] = X[q] - C*0.5*(1-theta)*(X[(q+1)%nx]**2 - X[(q-1)%nx]**2) \
                    + (D)*(1-theta)*(X[(q-1)%nx]-2*X[q]+X[(q+1)%nx]) \
                    -w[q] - C*0.5*theta*(w[(q+1)%nx]**2-w[(q-1)%nx]**2) \
                    + D*theta*(w[(q+1)%nx]-2*w[q] +w[(q-1)%nx])
            
                ### Create A^k_j, tridiag ###
                A = np.zeros([nx,nx])
                for p in range(1,nx):
                    A[p-1,p] = ((theta*C*w[(p+1)%nx])-(theta*D)) # down zero across 1 = b
                for p in range(0,nx-1):
                    A[p+1,p] = ((-theta*C*w[(p-1)%nx])-(theta*D)) # down 1 acros 0 = c
                for p in range(0,nx):
                    A[p,p] = (1+ 2*theta*D + C*theta*(w[(p+1)%nx]-w[(p-1)%nx]))
        
                ## solving for dw
                dw = scipy.linalg.solve(A, beta)
                err = np.linalg.norm(dw,2)
                ## improve the newton loop
                w = w + dw 
                ## we have w^k
            for j in range(0,nx):
                X[j] = w[j]

    return X 

def SolveBurger(nx ,nt ,Tfinal ,mu, method, theta, form):
    """SolveBurger: 
    
SUMMARY:
    This function takes an "initial_condition" and solves
    the "conservative" or "non conservative" burgers equation 
    numerically for some later time "Tfinal".
                    
FLEXIBILITY:
    You can choose:  the spatial discretisation used "nx", the number of timesteps "nt",
    the viscosity term "mu", the "method" used, the implicitness of the scheme
    "theta"
                    
FORMS OF BURGERS EQUATION:
    Non Conservative form  -->    u_t + u(u_x) - mu u_{xx} = 0 
    Conservative form      -->    u_t + (u^2/2)_x - mu u_{xx} = 0 
      
INPUT:      
    nx      = number of space points
    nt      = number of time points 
    Tfinal  = the final time  
    mu      = the viscosity in the burgers equation 
    method  = "picard" SOLVES APPROXIMATION TO BURGERS EQUATION
              "newton" SOLVES EXACT BURGERS EQUATION
    theta   = 1 is implicit,
              0 is explicit,
              You can put in other real values, 
    form    = "conservative"
              "non conservative"
Output:
    A vector representing the numerical solution 
                """
    
    ##**Creating the structure**## 
    X = np.zeros( [nx] )##**Each row contains a timestep**##
    x = np.linspace(0,1,nx)
    ### Initialisation of Initial conditions, as a d vector###
    X[:] = jump(x, 0.1, 0.3)

    
### Timestepping the whole scheme 
    if method == "newton":
        newton(X, nx, nt, Tfinal, mu, theta, form)
    if method == "picard":
        picard(X, nx, nt, Tfinal, mu, theta, form)
    return X

def Plotting(X,nx,Tfinal):
    '''Plotting:
        
SUMMARY:
    This function plots the solution vector obtained by SolveBurger():
    
FLEXIBILITY:
    The function has no flexibility when used correctly, 
    the three arguements should be the same as in SolveBurger.
    
INPUT: 
    X      = the solution obtained when peforming SolveBurger()
    nx     = the spatial discretisation
    Tfinal = the time you are plotting the solution 
    
OUTPUT:
    visual plot
    '''
    x = np.linspace(0,1,nx)
    y = X[:]
    plt.plot(x, y,linewidth=0.5, label = "t = %a"%Tfinal )
    plt.ylabel('$\phi$')
    plt.xlabel('x')
    plt.legend()
    

X = SolveBurger(301,205,0,0.001,"newton",1,"conservative")
Plotting(X,301,0)
X = SolveBurger(301,205,0.5,0.001,"newton",1,"conservative")
Plotting(X,301,0.5)



    
