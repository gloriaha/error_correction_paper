# import necessary functions

import numpy as np
import scipy.stats as st
import pandas as pd

def generate_bins(df, interval, t1, t2):
    """Generates dictionary with summary statistics for kinetochore counts for time bins of a specified width

    Parameters
    ----------
    df : DataFrame
        cleaned dataframe containing kinetochore counts and Mps1i addition - NEBD time info
    interval : int
        size of each bin in minutes
    t1 : int
        first Mps1i time - NEBD time to consider
    t2 : int
        last Mps1i time - NEBD time to consider (last bin is from t2 to t2+interval)

    Returns
    -------
    bin_dict : dict
        contains summary statistics for each bin
    """
    # initialize arrays
    hard_bin_means = []
    hard_bin_stds = []
    hard_bin_sems = []
    hard_times = np.linspace(t1,t2,int((t2-t1)//interval+1))
    sample_sizes = []
    
    # calculate binned summary statistics
    for z in hard_times:
        dNk2 = (df[(df['Mps1i time - NEBD time'].astype(int)<z+interval) & (df['Mps1i time - NEBD time'].astype(int)>=z)]['dNk'])**2
        hard_bin_means.append(np.mean(dNk2))
        hard_bin_stds.append(np.std(dNk2))
        hard_bin_sems.append(st.sem(dNk2))
        sample_sizes.append(len(dNk2))
    
    # compile everything in a dictionary
    bin_dict = dict({'times': np.array(hard_times)+(interval-1)/2, 'means': np.array(hard_bin_means),'stds':np.array(hard_bin_stds),'sems':np.array(hard_bin_sems),'plot_times':np.linspace(np.min(hard_times),np.max(hard_times)+1),
                   'n':sample_sizes})
    return bin_dict

def generate_dfs_from_file_mon(foldername, filename, times_list = ['5 min','10 min','15 min','20 min','25 min','30 min'],noAZname='no Mps1i'):
    """Generates cleaned dataframes and arrays describing kinetochore counts and timing of monastrol washout data

    Parameters
    ----------
    foldername : str
        name of folder
    filename : str
        name of CSV file
    times_list : list of str
        forced anaphase times to consider
    noAZname : str
        name of unforced 'AZ addition time (min)'

    Returns
    -------
    df_errorbar : DataFrame
        contains summary statistics for each time point
    anaphase_df : DataFrame
        contains cleaned data for unforced cells that have anaphase times
    anaphase_times : Series
        anaphase times
    unforced_dNk : Series
        unforced kinetochore count differences
    cleaned_df : DataFrame
        cleaned data (both forced and unforced)
    
    """
    # import and clean data
    imported_df = pd.read_csv(foldername+'/'+filename+'.csv')
    cleaned_df = imported_df.dropna(subset=['N1','N2','dNk'])
    cleaned_df['dNk']=pd.to_numeric(cleaned_df['dNk'],errors='coerce')
    cleaned_df['N1']=pd.to_numeric(cleaned_df['N1'],errors='coerce')
    cleaned_df['dNk2'] = cleaned_df['dNk']**2
    cleaned_df = cleaned_df.dropna(subset=['N1','N2','dNk'])
    cleaned_df= cleaned_df.astype({'dNk': 'float'})
    
    # extract data from each time
    AZtimes = times_list+[noAZname]
    cleandiffs = [cleaned_df[cleaned_df['AZ addition time (min)']==AZtime]['dNk'] for AZtime in AZtimes]
    df_errorbar = pd.DataFrame({'x':range(len(AZtimes)),
                                'AZ addition time':AZtimes,
                                'n':[len(diff) for diff in cleandiffs],
                                'mean':[np.mean(diff) for diff in cleandiffs],
                                'meansq':[np.mean(diff**2) for diff in cleandiffs],
                                'variance':[np.var(diff) for diff in cleandiffs],
                                'std':[np.std(diff) for diff in cleandiffs],
                                'stdsq':[np.std(diff**2) for diff in cleandiffs],
                                'sem':[st.sem(diff) for diff in cleandiffs],
                                'semsq':[st.sem(diff**2) for diff in cleandiffs]})
    
    # calculate unforced kinetochore counts
    unforced_dNk = cleaned_df[cleaned_df['AZ addition time (min)']==noAZname]['dNk']
    
    # make dataframe for unforced anaphase times
    anaphase_df = imported_df.dropna(subset=['Anaphase onset (min)'])
    anaphase_df = anaphase_df[anaphase_df['AZ addition time (min)']==noAZname]
    anaphase_df = anaphase_df[pd.to_numeric(anaphase_df['Anaphase onset (min)'],errors='coerce').notnull()]
    anaphase_times = anaphase_df['Anaphase onset (min)'].astype(int)
    return df_errorbar,anaphase_df,anaphase_times,unforced_dNk,cleaned_df

def generate_dfs_from_file_unsync(foldername,filename,noAZname='no Mps1i'):
    """Generates cleaned dataframes and arrays describing kinetochore counts and timing of unsynchronized data

    Parameters
    ----------
    foldername : str
        name of folder
    filename : str
        name of CSV file
    noAZname : str
        name of unforced 'AZ addition time (min)'

    Returns
    -------
    anaphase_df : DataFrame
        contains cleaned data for unforced cells that have anaphase times
    anaphase_times : Series
        anaphase times
    unforced_dNk : Series
        unforced kinetochore count differences
    cleaned_df : DataFrame
        cleaned data for forced anaphase cells
    
    """
    # import and clean data
    imported_df = pd.read_csv(foldername+'/'+filename+'.csv')
    
    # make dataframe for unforced anaphase times
    anaphase_df = imported_df.dropna(subset=['NEBD time','anaphase onset time','anaphase onset time - NEBD time'])
    anaphase_df = anaphase_df[anaphase_df['AZ addition time (min)']==noAZname]
    anaphase_df = anaphase_df[pd.to_numeric(anaphase_df['NEBD time'],errors='coerce').notnull()]
    anaphase_df = anaphase_df[pd.to_numeric(anaphase_df['anaphase onset time - NEBD time'],errors='coerce').notnull()]
    anaphase_times = anaphase_df['anaphase onset time - NEBD time'].astype(int)
    
    # make dataframe for unforced kinetochore counts
    unforced_df=imported_df.dropna(subset=['N1','N2','dNk'])
    unforced_df=unforced_df[unforced_df['AZ addition time (min)']==noAZname]
    unforced_df['dNk'] = pd.to_numeric(unforced_df['dNk'],errors='coerce')
    unforced_df = unforced_df.dropna(subset=['dNk'])
    unforced_dNk=unforced_df['dNk'].astype(int)
    
    # make dataframe for forced anaphase data
    cleaned_df = imported_df.dropna(subset=['N1','N2','dNk','Mps1i time - NEBD time'])
    cleaned_df['dNk'] = pd.to_numeric(cleaned_df['dNk'],errors='coerce')
    cleaned_df = cleaned_df.dropna(subset=['dNk'])
    cleaned_df['Mps1i time - NEBD time'] = pd.to_numeric(cleaned_df['Mps1i time - NEBD time'],errors='coerce')
    cleaned_df = cleaned_df.dropna(subset=['Mps1i time - NEBD time'])
    cleaned_df = cleaned_df[cleaned_df['Mps1i time - NEBD time']>-1]
    cleaned_df['N1'] = pd.to_numeric(cleaned_df['N1'],errors='coerce')
    cleaned_df = cleaned_df.dropna(subset=['N1'])
    cleaned_df['N1']=cleaned_df['N1'].astype(int)
    cleaned_df = cleaned_df[cleaned_df['N1']!='']

    return anaphase_df,anaphase_times,cleaned_df,unforced_df,unforced_dNk

def generate_bins_mad2(ana_times,dNk2s, interval, t1, t2):
    """Generates dictionary with summary statistics for kinetochore counts for time-binned Mad2 data

    Parameters
    ----------
    ana_times : Series
        anaphase times
    dNk2s : Series
        squared kinetochore counts corresponding to anaphase times
    interval : int
        size of each bin in minutes
    t1 : int
        first anaphase time to consider
    t2 : int
        last anaphase time to consider (last bin is from t2 to t2+interval)

    Returns
    -------
    bin_dict : dict
        contains summary statistics for each bin
    """
    # initialize arrays
    hard_bin_means = []
    hard_bin_stds = []
    hard_bin_sems = []
    hard_times = np.linspace(t1,t2,int((t2-t1)//interval+1))
    sample_sizes = []
    
    # calculate summary statistics for bins
    for z in hard_times:
        dNk2 = dNk2s[(ana_times.astype(int)<z+interval) & (ana_times.astype(int)>=z)]
        hard_bin_means.append(np.mean(dNk2))
        hard_bin_stds.append(np.std(dNk2))
        hard_bin_sems.append(st.sem(dNk2))
        sample_sizes.append(len(dNk2))
        
    # compile everything in a dictionary
    bin_dict = dict({'times': np.array(hard_times)+(interval-1)/2, 'means': np.array(hard_bin_means),'stds':np.array(hard_bin_stds),'sems':np.array(hard_bin_sems),'plot_times':np.linspace(np.min(hard_times),np.max(hard_times)+1),
                   'n':sample_sizes})
    return bin_dict 