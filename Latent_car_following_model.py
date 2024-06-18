import numpy as np
import pandas as pd
from scipy.stats import norm, t # normal distribution density function

import helpers
import optmizer

def Latent_car_following_model(
          data,
          t_reac,
          model_name,
          panel=1,
          mixing=0,
          remove_variables = {},
          verbose=2,
          betas_start = None,
          pbar=None
):
    # Data-processing
    data['running_task'] = data.groupby(['ID']).cumcount()+1 # counter of observations per individual

    Acceleration = np.array(data['Acceleration']).reshape(-1, 1)
    Speed = np.array(data['Speed']).reshape(-1, 1)
    Space_headway = np.array(data[f'lag_s_headway{t_reac}']).reshape(-1, 1)
    Speed_diff = np.array(data[f'lag_speed{t_reac}']).reshape(-1, 1) - np.array(data[f'lag_speed_lead{t_reac}']).reshape(-1, 1)

    ID = np.array(data['ID']) # ID does not need to be reshaped

    # Define betas if not provided
    if betas_start is None:
        betas_start = {
            "bc_acc":0, # Constant of acceleration utility
            "bc_dec":0, # Constant of deceleration utility

            "b_speed_c1":0, # Relative speed for acceleration utility
            "b_speed_c2":0, # Relative speed for deceleration utility

            "b_space_headway_c1":0, # Relative space_headway for acceleration utility
            "b_space_headway_c2":0, # Relative space_headway for deceleration utility

            "alpha_acc":0, # Acceleration constant parameter
            "alpha_dec":0, # Deceleration constant parameter
            "beta_acc":0,  # Speed (acceleration regime) parameter
            "beta_dec":0,  # Speed (deceleration regime) parameter
            "gamma_acc":0, # Space headway (acceleration regime) parameter
            "gamma_dec":0, # Space headway (deceleration regime) parameter
            "lamda_acc_p":0, # Relative speed (acceleration regime) parameter
            # "lamda_acc_n":0, # Relative speed (deceleration regime) parameter
            # "lamda_dec_p":0, # Relative speed (acceleration regime) parameter
            "lamda_dec_n":0, # Relative speed (deceleration regime) parameter
            "dn":0.0,
            "sigma_acc":-0, # Std deviation (acceleration regime) parameter
            "sigma_dec":-0,  # Std deviation (deceleration regime) parameter
            "sigma_dn":-0  # Std deviation (deceleration regime) parameter
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

        ############################################################################################################
        ############################################################################################################
        
        # Latent classes
        
        # Utility functions
        U_acc = bc_acc + b_speed_c1*Speed_diff*(Speed_diff>0) + b_space_headway_c1*Space_headway
        U_dec = bc_dec + b_speed_c2*Speed_diff*(Speed_diff<0) + b_space_headway_c2*Space_headway
        U_dn = 0

        # Utility exponentials
        U_acc = np.exp(U_acc)
        U_dec = np.exp(U_dec)
        U_dn = np.exp(U_dn)
        
        # Sum of utilities
        sum_all = U_acc + U_dec + U_dn
        
        # Probabilities
        P_acc = U_acc/sum_all
        P_dec = U_dec/sum_all
        P_dn = U_dn/sum_all
        
        ############################################################################################################
        ############################################################################################################
        
        # Car-following model specification

        # BE CAREFUL!! Below we add a small correction term (np.exp(-50)) to speed and relative speed.
        # This correction is included to avoid estimation issues if the value of the independent variables is 0.
        
        # BE CAREFUL!! We force the signs of alpha_acc and alpha_dec by using then inside an exp() funciton.
        
        # Sensitivity term
        sensitivity_acc = np.exp(alpha_acc)*((Speed+np.exp(-50))**beta_acc)/(Space_headway**gamma_acc)
        sensitivity_dec = -np.exp(alpha_dec)*((Speed+np.exp(-50))**beta_dec)/(Space_headway**gamma_dec)

        # Stimulus term (we add a condition to keep only the positive values for acceleration and negative for deceleration)
        stimulus_acc = (
            (np.abs(Speed_diff + np.exp(-50))**lamda_acc_p)**(Speed_diff>=0)
            # (np.abs(Speed_diff + np.exp(-50))**lamda_acc_n)*(Speed_diff<0)
            )
        
        stimulus_dec = (
            # (np.abs(Speed_diff + np.exp(-50))**lamda_dec_p)*(Speed_diff>=0)+
            (np.abs(Speed_diff + np.exp(-50))**lamda_dec_n)**(Speed_diff<0)
            )

        # Acceleration - deceleration means
        acc = sensitivity_acc*stimulus_acc
        dec = sensitivity_dec*stimulus_dec

        # Density functions for acceleration and deceleration
        pdf_acc = norm.pdf(Acceleration,acc,np.exp(sigma_acc))/(1-norm.cdf(0,acc,np.exp(sigma_acc)))
        pdf_dec = norm.pdf(Acceleration,dec,np.exp(sigma_dec))/(norm.cdf(0,dec,np.exp(sigma_dec)))
        
        P_acc_dn = 1-norm.cdf(0, dn, np.exp(sigma_dn))
        P_dec_dn = norm.cdf(0, dn, np.exp(sigma_dn))

        pdf_dn = norm.pdf((Acceleration),dn,np.exp(sigma_dn))
        
        #Acceleration and deceleration total probability
        f_acc = (pdf_acc*P_acc+pdf_dn*P_dn) / (P_acc + P_acc_dn * P_dn)
        f_dec = (pdf_dec*P_dec+pdf_dn*P_dn) / (P_dec + P_dec_dn * P_dn)
        # Combined probability of acceleration and deceleration regimes (considering do nothing state as well)
        P = (f_acc*(Acceleration>=0) + f_dec*(Acceleration<0))
        
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
        pbar=pbar #Because it had trouble converging
    )

    # Return the result and a dict with other usefull vars (to compute results and goodness of fit)
    return (result,
            {
                 "betas_start": betas_start,
                 "SLL": SLL,
                 "LL": LL,
                 "Nobs": data.shape[0],
            })