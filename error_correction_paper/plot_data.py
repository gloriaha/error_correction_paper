# import necessary functions
import numpy as np
import scipy.stats as st
import pandas as pd
from scipy.optimize import curve_fit
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import linalg
import matplotlib.pyplot as plt
from functools import partial


# set total number of chromosomes to be 46
c_tot = 46

def symm_expo_fit(x,A,kb,cb):
    """Calculates exponential model prediction with statistical symmetry

    Parameters
    ----------
    x : ndarray
        times to consider in minutes
    A : float
        kb/(kb+ke)
    kb : float
        correction rate (units of 1/min)
    cb : float
        initial fraction of correct attachments

    Returns
    -------
    expo : ndarray
        model prediction given parameters
    """
    ke = (kb)*(1-A)/(A)
    expo = 4*c_tot*((1-A)+(A-cb)*np.exp(-(ke+kb)*(x)))
    return expo

def plot_expo_with_error_unp(bin_dict,colorname='g',labelname='data',p0=[0.5,0.1,0.5]):
    """Fits and plots exponential model with statistical symmetry

    Parameters
    ----------
    bin_dict : dict
        summary statistics of kinetochore counts in time bins
    colorname : str
        color code for plotting
    labelname : str
        condition name to use in plot labels
    p0 : list
        initial guess for A, kb, and cb
    """
    x = bin_dict['times']
    y = bin_dict['means']
    
    # fit data to single exponential
    popt, pcov = curve_fit(symm_expo_fit, x, y,p0=p0,sigma=bin_dict['stds'],bounds=((0, 0, 0), (1, 1, 1)))
    A,kb,cb = unc.correlated_values(popt, pcov)
    
    # calculate uncertainty on fit
    px = bin_dict['plot_times']
    ke = (kb)*(1-A)/(A)
    py =  4*c_tot*((1-A)+(A-cb)*unp.exp(-(ke+kb)*(px)))
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)
    
    # plot data, fit, and uncertainty
    plt.errorbar(x=bin_dict['times'],y=bin_dict['means'],yerr=bin_dict['sems'],fmt='o',color=colorname,linewidth=3,markersize=9,label=labelname)
    plt.semilogy(bin_dict['plot_times'],symm_expo_fit(bin_dict['plot_times'],*popt), '--',color=colorname, linewidth=5,alpha=0.25)
    plt.fill_between(bin_dict['plot_times'],nom-2*std,nom+2*std,color=colorname,alpha=0.1)
    plt.ylim([10e-2,np.max(y)+300])
    plt.xlabel('Mps1i addition time - NEBD time (min)')
    plt.ylabel(r'$\langle(\Delta N)^2\rangle$')
    
    # print fit results
    print(labelname)
    print('k_b',kb)
    print('A',A)
    print('cb',cb)
    print('ke',ke)
    
def full_gumbel_offset_pdf(onset_times,cb,kb,toff):
    """Calculates Gumbel model prediction for anaphase times

    Parameters
    ----------
    onset_times : ndarray
        times to consider in minutes
    cb : float
        initial fraction of correct attachments
    kb : float
        correction rate (units of 1/min)
    toff : float
        offset time (min)

    Returns
    -------
    result : ndarray
        model prediction given parameters
    """
    e_i = c_tot*(1-cb)
    if cb>1 or cb<0:
        result = np.inf
        return result
    else:
        result = kb*e_i*np.exp(-e_i*np.exp(-kb*(onset_times-toff))-kb*(onset_times-toff))/(1-np.exp(-e_i))
        return result

def plot_only_symm_gumbel(anaphase_times,colorname='g',labelname='data',p0=[0.5,0.5,5],confidence=True):
    """Fits and plots Gumbel model for anaphase times

    Parameters
    ----------
    anaphase times : Series
        anaphase onset times (min)
    colorname : str
        color code for plotting
    labelname : str
        condition name to use in plot labels
    p0 : list
        initial guess for cb, kb, and toff
    """
    # prepare anaphase time data
    nbins = int((np.unique(anaphase_times)[-1]-np.unique(anaphase_times)[0]))
    hist,bin_edges = np.histogram(anaphase_times,bins=nbins)
    hist=hist/sum(hist)
    n = len(hist)
    x_hist=np.zeros((n),dtype=float)
    for ii in range(n):
        x_hist[ii]=(bin_edges[ii+1]+bin_edges[ii])/2
        
    # fit anaphase times to model
    popt, pcov = curve_fit(full_gumbel_offset_pdf, x_hist, hist,p0=p0)
    cb,kb,toff = unc.correlated_values(popt, pcov)
    
    # plot data and model fit
    plt.hist(anaphase_times,bins=nbins,density=True,color=colorname,label='data (n='+str(len(anaphase_times))+')',alpha=0.5)
    plt.plot(np.linspace(0,np.max(anaphase_times)+10,1000),full_gumbel_offset_pdf(np.linspace(0,np.max(anaphase_times)+10,1000), popt[0],popt[1],popt[2]),'--',color=colorname,linewidth=3,label='model fit')    
    plt.xlabel('anaphase onset time (min)')
    plt.ylabel('fraction of cells')
    
    # plot uncertainty
    if confidence:
        px = np.linspace(0,np.max(anaphase_times)+10,1000)
        e_i = c_tot*(1-cb)
        py = kb*e_i*unp.exp(-e_i*unp.exp(-kb*(px-toff))-kb*(px-toff))/(1-unp.exp(-e_i))
        nom = unp.nominal_values(py)
        std = unp.std_devs(py)
        plt.fill_between(px,nom-2*std,nom+2*std,color='k',alpha=0.5)
        
    # print results
    print(labelname)
    print('k_b',kb)
    print('cb',cb)
    print('t_off',toff)
    
def asymm_expo_offset_fit(params,xvals):
    """Calculates exponential model prediction with statistical asymmetry

    Parameters
    ----------
    params : list
        cb, kb, ke, toff, asym
    xvals : ndarray
        times to consider (min)

    Returns
    -------
    expo_vals : ndarray
        model prediction given parameters
    """
    cb,kb,ke,toff,asym=params
    A = kb/(kb+ke)
    expo_vals = 4*c_tot*(1-A)+4*c_tot*(A-cb)*np.exp(-(kb+ke)*xvals)+4*np.exp(-2*kb*xvals)*asym
    return expo_vals

def symm_expo_offset_fit(params,xvals):
    """Calculates exponential model prediction with statistical symmetry

    Parameters
    ----------
    params : list
        cb, kb, ke, toff, asym
    xvals : ndarray
        times to consider (min)

    Returns
    -------
    expo_vals : ndarray
        model prediction given parameters
    """
    cb,kb,ke,toff=params
    A = kb/(kb+ke)
    expo_vals = 4*c_tot*(1-A)+4*c_tot*(A-cb)*np.exp(-(kb+ke)*(xvals))
    return expo_vals

def piecewise_symm_fullnorm_func(data,cb,kb,ke,toff,y_fullnorm):
    """Calculates simultaneous fit predictions with statistical symmetry

    Parameters
    ----------
    data : ndarray
        concatenated binned kinetochore count and anaphase time data
    cb : float
        initial fraction of correct attachments
    kb : float
        correction rate (units of 1/min)
    ke : float
        error rate (units of 1/min)
    toff : float
        offset time (min)

    Returns
    -------
    result : ndarray
        model prediction given parameters
    """
    A = kb/(kb+ke)
    expo_vals = (4*c_tot*(1-A)+4*c_tot*(A-cb)*np.exp(-(kb+ke)*(data[:y_fullnorm])))/y_fullnorm
    gumbel_vals=full_gumbel_offset_pdf(data[y_fullnorm:],cb,kb,toff)
    result = np.concatenate((expo_vals,gumbel_vals))
    return result


def plot_symm_gumbel_fullnorm_piecewise(bin_dict,anaphase_times,colorname='g',labelname='data',shape='o',p0=[0.5,0.5,0.001,10],which_plot='expo',alpha=1,confidence=True):
    """Performs and plots simultaneous fit with statistical asymmetry

    Parameters
    ----------
    bin_dict : dict
        summary statistics of kinetochore counts in time bins
    anaphase times : Series
        anaphase onset times (min)
    colorname : str
        color code for plotting
    labelname : str
        condition name to use in plot labels
    shape : str
        shape code for plotting
    p0 : list
        initial guess for cb, kb, ke, toff
    which_plot : str
        which plot to show
    alpha : float
        transparency for plotting
    confidence : Boolean
        whether or not to plot uncertainty
    """
    # prepare data
    x = bin_dict['times']
    y = bin_dict['means']
    y_fullnorm = len(x)
    nbins = int((np.unique(anaphase_times)[-1]-np.unique(anaphase_times)[0]))
    hist,bin_edges = np.histogram(anaphase_times,bins=nbins)
    hist=hist/sum(hist)
    n = len(hist)
    x_hist=np.zeros((n),dtype=float)
    for ii in range(n):
        x_hist[ii]=(bin_edges[ii+1]+bin_edges[ii])/2
    y_normed = y/y_fullnorm
    x_full = np.concatenate((x,x_hist))
    y_full = np.concatenate((y_normed,hist))
    sigma_full = np.concatenate((bin_dict['stds'],np.ones_like(x_hist)))
    
    # fit data to model
    popt, pcov = curve_fit(partial(piecewise_symm_fullnorm_func,y_fullnorm=y_fullnorm), x_full, y_full, p0=p0,sigma=sigma_full,bounds=((0, 0, 0,0), (1, 1, 1,50)))
    cb,kb,ke,toff = unc.correlated_values(popt, pcov)
    
    # plot exponential fit
    if which_plot=='expo':
        px = bin_dict['plot_times']
        A = kb/(kb+ke)
        py = 4*c_tot*(1-A)+4*c_tot*(A-cb)*unp.exp(-(kb+ke)*(px))
        nom = unp.nominal_values(py)
        std = unp.std_devs(py)
        plt.semilogy(bin_dict['plot_times'],symm_expo_offset_fit(popt,bin_dict['plot_times']), '--',color=colorname, linewidth=5,alpha=alpha*0.25)
        plt.errorbar(x=bin_dict['times'],y=bin_dict['means'],yerr=bin_dict['sems'],fmt=shape,color=colorname,linewidth=3,markersize=9,label=labelname,alpha=alpha)
        if confidence:
            plt.fill_between(bin_dict['plot_times'],nom-2*std,nom+2*std,color=colorname,alpha=alpha*0.1)
        plt.ylim([10e-2,np.max(y)+300])
        plt.legend();
        plt.xlabel('Mps1i addition time - NEBD time (min)')
        plt.ylabel(r'$\langle(\Delta N)^2\rangle$')
    
    # plot anaphase time fit
    elif which_plot=='gumbel':
        px = np.linspace(0,np.max(anaphase_times)+10,1000)
        e_i = c_tot*(1-cb)
        py = kb*e_i*unp.exp(-e_i*unp.exp(-kb*(px-toff))-kb*(px-toff))/(1-unp.exp(-e_i))
        nom = unp.nominal_values(py)
        std = unp.std_devs(py)
        plt.hist(anaphase_times,bins=nbins,density=True,alpha=0.5*alpha,color=colorname,label=labelname)
        if confidence:
            plt.fill_between(px,nom-2*std,nom+2*std,color='k',alpha=0.25)
        plt.plot(px,full_gumbel_offset_pdf(px, popt[0],popt[1],popt[3]),'--',linewidth=2,color=colorname)
        plt.xlabel('anaphase onset time (min)')
        plt.ylabel('fraction of cells')
    # plot both exponential and anaphase time fit
    else:
        px = bin_dict['plot_times']
        A = kb/(kb+ke)
        py = 4*c_tot*(1-A)+4*c_tot*(A-cb)*unp.exp(-(kb+ke)*(px))
        nom = unp.nominal_values(py)
        std = unp.std_devs(py)
        plt.semilogy(bin_dict['plot_times'],symm_expo_offset_fit(popt,bin_dict['plot_times']), '--',color=colorname, linewidth=5,alpha=alpha*0.25)
        plt.errorbar(x=bin_dict['times'],y=bin_dict['means'],yerr=bin_dict['sems'],fmt=shape,color=colorname,linewidth=3,markersize=9,label=labelname,alpha=alpha)
        if confidence:
            plt.fill_between(bin_dict['plot_times'],nom-2*std,nom+2*std,color=colorname,alpha=alpha*0.1)
        plt.ylim([10e-2,np.max(y)+300])
        plt.legend();
        plt.xlabel('Mps1i addition time - NEBD time (min)')
        plt.ylabel(r'$\langle(\Delta N)^2\rangle$')
        plt.figure();
        px = np.linspace(0,np.max(anaphase_times)+10,1000)
        e_i = c_tot*(1-cb)
        py = kb*e_i*unp.exp(-e_i*unp.exp(-kb*(px-toff))-kb*(px-toff))/(1-unp.exp(-e_i))
        nom = unp.nominal_values(py)
        std = unp.std_devs(py)
        plt.hist(anaphase_times,bins=nbins,density=True,alpha=0.5*alpha,color=colorname,label=labelname)
        if confidence:
            plt.fill_between(px,nom-2*std,nom+2*std,color='k',alpha=0.25)
        plt.plot(px,full_gumbel_offset_pdf(px, popt[0],popt[1],popt[3]),'--',linewidth=2,color=colorname)
        plt.xlabel('anaphase onset time (min)')
        plt.ylabel('fraction of cells')
        
    # print results
    print(labelname)
    print('k_b',kb)
    print('ke',ke)
    print('cb',cb)
    print('t_off',toff)
    print('CEinit',(1-cb)*c_tot)
    

def gumbel_with_ke_toff_binom_fit_pdf(times, cb, kb, ke, toff):
    """Calculates Gumbel model prediction for anaphase times with finite ke

    Parameters
    ----------
    times : ndarray
        anaphase times to consider in minutes
    cb : float
        initial fraction of correct attachments
    kb : float
        correction rate (units of 1/min)
    toff : float
        offset time (min)

    Returns
    -------
    P : ndarray
        model prediction given parameters
    """
    # define
    N = 46
    A = np.zeros((N,N)) # states 0 to N-1, transition matrix
    v = st.binom.pmf(range(0,46),46,cb) # states 0 to N-1, vector of initial condition
    for j in range(1,N-1):
        A[j,j+1] = ke*(j+1)
        A[j,j-1] = kb*(N-j+1)
        A[j,j] = -ke*(j) - kb*(N-j)
    A[0,1] = ke;
    A[N-1,N-2] = 2*kb
    A[0,0] = -A[1,0]
    A[N-1,N-1] = -A[N-2,N-1]-kb
    dt = 1
    time_array = np.arange(min(times),max(times)+2,dt)
    Res = np.zeros(len(time_array))
    for i,t in enumerate(time_array-toff):
        Res[i] = np.sum(np.matmul(linalg.expm(A*t),v))
    P = (-Res[1:]+Res[0:len(Res)-1])/dt
    return P
    
def piecewise_asymm_gumbel_ke_binom_func(data,cb,kb,ke,toff,asymm,y_fullnorm):
    """Calculates simultaneous fit predictions with statistical asymmetry and finite ke in SFPT

    Parameters
    ----------
    data : ndarray
        concatenated binned kinetochore count and anaphase time data
    cb : float
        initial fraction of correct attachments
    kb : float
        correction rate (units of 1/min)
    ke : float
        error rate (units of 1/min)
    toff : float
        offset time (min)
    asymm : float
        asymmetry amplitude

    Returns
    -------
    result : ndarray
        model prediction given parameters
    """
    A = kb/(kb+ke)
    expo_vals = (4*c_tot*(1-A)+4*c_tot*(A-cb)*np.exp(-(kb+ke)*(data[:y_fullnorm]))+4*asymm*np.exp(-2*kb*data[:y_fullnorm]))/y_fullnorm
    gumbel_vals=gumbel_with_ke_toff_binom_fit_pdf(data[y_fullnorm:],cb,kb,ke,toff)
    result = np.concatenate((expo_vals,gumbel_vals))
    return result

def plot_asymm_gumbel_ke_binom_piecewise(bin_dict,anaphase_times,colorname='g',labelname='data',p0=[0.1,0.15,0.01,5,100],shape='o',which_plot='both',confidence=True):
    """Performs and plots simultaneous fit with statistical symmetry and finite ke in slowest first passage time
    
    Parameters
    ----------
    bin_dict : dict
        summary statistics of kinetochore counts in time bins
    anaphase times : Series
        anaphase onset times (min)
    colorname : str
        color code for plotting
    labelname : str
        condition name to use in plot labels
    shape : str
        shape code for plotting
    p0 : list
        initial guess for cb, kb, ke, toff, asymm
    which_plot : str
        which plot to show
    alpha : float
        transparency for plotting
    confidence : Boolean
        whether or not to plot uncertainty
    """
    # prepare data
    x = bin_dict['times']
    y = bin_dict['means']
    y_fullnorm = len(x)
    nbins = int((np.unique(anaphase_times)[-1]-np.unique(anaphase_times)[0]))
    hist,bin_edges = np.histogram(anaphase_times,bins=nbins)
    hist=hist/sum(hist)
    n = len(hist)
    x_hist=np.zeros((n),dtype=float)
    for ii in range(n):
        x_hist[ii]=(bin_edges[ii+1]+bin_edges[ii])/2
    x_full = np.concatenate((x,x_hist))
    y_full = np.concatenate((y/y_fullnorm,hist))
    sigma_full = np.concatenate((bin_dict['stds'],np.ones_like(x_hist)))
    
    # fit data to model
    popt, pcov = curve_fit(partial(piecewise_asymm_gumbel_ke_binom_func,y_fullnorm=y_fullnorm), x_full, y_full,p0=p0,sigma=sigma_full,bounds=((0, 0, 0,0,0), (1, 1, 1,50,1000)))
    cb,kb,ke,toff,asymm = unc.correlated_values(popt, pcov)
    
    # plot exponential fit
    if which_plot=='expo':
        plt.semilogy(bin_dict['plot_times'],asymm_expo_offset_fit(popt,bin_dict['plot_times']), '--',color=colorname, linewidth=5,alpha=0.25)
        plt.errorbar(x=bin_dict['times'],y=bin_dict['means'],yerr=bin_dict['sems'],fmt=shape,color=colorname,linewidth=3,markersize=9,label=labelname)
        if confidence:
            px = bin_dict['plot_times']
            A = kb/(kb+ke)
            py = 4*c_tot*(1-A)+4*c_tot*(A-cb)*unp.exp(-(kb+ke)*(px))+4*asymm*unp.exp(-2*kb*px)
            nom = unp.nominal_values(py)
            std = unp.std_devs(py)
            plt.fill_between(bin_dict['plot_times'],nom-2*std,nom+2*std,color=colorname,alpha=0.1)
        plt.ylim([10e-2,np.max(y)+300])
        plt.legend();
        plt.xlabel('Mps1i addition time - NEBD time (min)')
        plt.ylabel(r'$\langle(\Delta N)^2\rangle$')
    
    # plot anaphase time fit
    elif which_plot=='gumbel':
        px = np.arange(int(popt[3]),np.max(anaphase_times)+10,1)
        plt.hist(anaphase_times,bins=nbins,density=True,alpha=0.5,color=colorname,label=labelname)
        plt.plot(px,gumbel_with_ke_toff_binom_fit_pdf(px, popt[0],popt[1],popt[2],popt[3]),'--',linewidth=2,color=colorname)
        
    # plot both exponential and anaphase time fit
    else:
        # plot exponential fit
        px = bin_dict['plot_times']
        A = kb/(kb+ke)
        py = 4*c_tot*(1-A)+4*c_tot*(A-cb)*unp.exp(-(kb+ke)*(px))+4*asymm*unp.exp(-2*kb*px)
        nom = unp.nominal_values(py)
        std = unp.std_devs(py)
        plt.semilogy(bin_dict['plot_times'],asymm_expo_offset_fit(popt,bin_dict['plot_times']), '--',color=colorname, linewidth=5,alpha=0.25)
        plt.errorbar(x=bin_dict['times'],y=bin_dict['means'],yerr=bin_dict['sems'],fmt=shape,color=colorname,linewidth=3,markersize=9,label=labelname)
        if confidence:
            plt.fill_between(bin_dict['plot_times'],nom-2*std,nom+2*std,color=colorname,alpha=0.1)
        plt.ylim([10e-2,np.max(y)+300])
        plt.legend();
        plt.xlabel('Mps1i addition time - washout time (min)')
        plt.ylabel(r'$\langle(\Delta N)^2\rangle$')
        
        # plot anaphase time fit
        plt.figure();
        px = np.arange(int(popt[3]),np.max(anaphase_times)+10,1)
        plt.hist(anaphase_times,bins=nbins,density=True,alpha=0.5,color=colorname,label=labelname)
        plt.plot(px,gumbel_with_ke_toff_binom_fit_pdf(px, popt[0],popt[1],popt[2],popt[3]),'--',linewidth=2,color=colorname)
        plt.xlabel('anaphase onset time')
        plt.ylabel('fraction of cells')
    # print results
    print(labelname)
    print('k_b',kb)
    print('ke',ke)
    print('cb',cb)
    print('t_off',toff)
    print('asymm',asymm)
    return popt