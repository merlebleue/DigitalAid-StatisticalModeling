import numpy as np
import pandas as pd
from scipy.stats import norm, t # normal distribution density function

import helpers
import optmizer

def Helly_car_following_model(
          data,
          t_reac,
          model_name,
          panel=1,
          mixing=0,
          remove_variables = {},
          verbose=2,
          pbar=None
):
    # Data-processing
    data['running_task'] = data.groupby(['ID']).cumcount()+1 # counter of observations per individual

    Acceleration = np.array(data['Acceleration']).reshape(-1, 1)
    Speed = np.array(data[f'lag_speed{t_reac}']).reshape(-1, 1)
    Space_headway = np.array(data[f'lag_s_headway{t_reac}']).reshape(-1, 1)
    Speed_diff = np.array(data[f'lag_speed{t_reac}']).reshape(-1, 1) - np.array(data[f'lag_speed_lead{t_reac}']).reshape(-1, 1)

    ID = np.array(data['ID']) # ID does not need to be reshaped

    # Define betas
    betas_start = {
         "alpha1":0,
         "alpha2":0,
         "beta1":0,
         "beta2":0,
         "sigma":0,
         }
    
    # Remove the keys that are in remove_variable
    {betas_start.pop(i) for i in remove_variables}
    
    # Define Log-likelihood function
    def LL(betas): # betas is a vector with the parameters we want to estimate
        # First let's define the parameters to be estimated.
        # The parameter names are imported directly from 'beta_start' that we defined earlier
        
        for pn in range(len(betas_start.values())):
            globals()[np.array(list(betas_start.keys()))[pn]] = betas[pn]

        for not_a_variable_anymore in remove_variables: #Input a default value so that the code works
            globals()[not_a_variable_anymore] = 0

        DSpace = beta1 + beta2*Speed
        
        fi = alpha1*Speed_diff + alpha2 * (Space_headway - DSpace)

        # Density function
        P = norm.pdf(Acceleration,fi,np.exp(sigma))

        ############################################################################################################
        # - Now this below is relevant if we have panel data and apply mixing (Do not change this piece of code!) -#
        if mixing == 1:
            # We take the average per row to get the average probability per individual (if mixing == 1)
                P = pd.DataFrame(P)
                P = pd.concat([pd.Series(ID), pd.DataFrame(P)], axis=1, ignore_index=True)
                P.rename(columns={P.columns[0]: 'ID'},inplace=True)
        
                # We take the product of probabilities per individual per draw and then delete the ID column
                P = P.groupby('ID', as_index=False).prod()
                P = P.drop('ID', axis=1)
                P['mean'] = P.mean(axis=1)
                P = np.array(P['mean'])

        elif panel == 1:
        # Do it as panel
            P = pd.DataFrame(P)
            P = pd.concat([pd.Series(ID), pd.DataFrame(P)], axis=1, ignore_index=True)
            P.rename(columns={P.columns[0]: 'ID'},inplace=True)
        
            # We take the product of probabilities per individual per draw and then delete the ID column
            P = P.groupby('ID', as_index=False).prod()
            P = P.drop('ID', axis=1)
                
        P = np.array(P)
        ### --- This is where the panel data approach ends. --- ###
        ############################################################################################################
        
        # We then take the log of the density function
        logprob = np.log(P)
        
        return logprob

    def SLL(betas):
        return -sum(LL(betas))
    #optimize

    result = optmizer.optimizer(
        betas_start=betas_start,
        SLL = SLL,
        model_name=model_name,
        verbose = verbose,
        pbar=pbar
    )

    # Return the result and a dict with other usefull vars (to compute results and goodness of fit)
    return (result,
            {
                 "betas_start": betas_start,
                 "SLL": SLL,
                 "LL": LL,
                 "Nobs": data.shape[0],
            })