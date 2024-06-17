import numpy as np
import pandas as pd
from scipy.stats import norm, t # normal distribution density function

import helpers
import optmizer

def GM_car_following_model(
          data,
          model_name,
          panel=1,
          mixing=0,
          remove_variables = {},
          verbose=2
):
    # Data-processing
    data['running_task'] = data.groupby(['ID']).cumcount()+1 # counter of observations per individual

    Acceleration = np.array(data['Acceleration']).reshape(-1, 1)
    Speed = np.array(data['Speed']).reshape(-1, 1)
    Space_headway = np.array(data['Space_headway']).reshape(-1, 1)
    Speed_diff = np.array(data['Speed_diff']).reshape(-1, 1)

    ID = np.array(data['ID']) # ID does not need to be reshaped

    # Define betas
    betas_start = {
                "alpha_acc":0, # Acceleration constant parameter
                "alpha_dec":0, # Deceleration constant parameter
                "beta_acc":0,  # Speed (acceleration regime) parameter
                "beta_dec":0,  # Speed (deceleration regime) parameter
                "gamma_acc":0, # Space headway (acceleration regime) parameter
                "gamma_dec":0, # Space headway (deceleration regime) parameter
                "lamda_acc":0, # Relative speed (acceleration regime) parameter
                "lamda_dec":0, # Relative speed (deceleration regime) parameter
                "sigma_acc":0, # Std deviation (acceleration regime) parameter
                "sigma_dec":0  # Std deviation (deceleration regime) parameter
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

        # Car-following model specification

        # BE CAREFUL!! Below we add a small correction term (np.exp(-50)) to speed and relative speed.
        # This correction is included to avoid estimation issues if the value of the independent variables is 0.
        
        # Sensitivity term
        sensitivity_acc = alpha_acc*((Speed+np.exp(-50))**beta_acc)/(Space_headway**gamma_acc)
        sensitivity_dec = alpha_dec*((Speed+np.exp(-50))**beta_dec)/(Space_headway**gamma_dec)

        # Stimulus term
        stimulus_acc = np.abs(Speed_diff + np.exp(-50))**lamda_acc
        stimulus_dec = np.abs(Speed_diff + np.exp(-50))**lamda_dec

        # Acceleration - deceleration means
        acc = sensitivity_acc*stimulus_acc
        dec = sensitivity_dec*stimulus_dec

        # Density functions for acceleration and deceleration
        pdf_acc = norm.pdf(Acceleration,acc,np.exp(sigma_acc))
        pdf_dec = norm.pdf(Acceleration,dec,np.exp(sigma_dec))
        
        # Combined probability of acceleration and deceleration regimes
        P = pdf_acc*(Speed_diff>=0)+pdf_dec*(Speed_diff<0)
        
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
        verbose = verbose
    )

    # Return the result and a dict with other usefull vars (to compute results and goodness of fit)
    return (result,
            {
                 "betas_start": betas_start,
                 "SLL": SLL,
                 "LL": LL,
                 "Nobs": data.shape[0],
            })