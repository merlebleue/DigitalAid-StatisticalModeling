import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Some constants
# VARIABLES
ACCELERATION = "Acceleration"
SPEED = "Speed"
SPEED_DIFF = "Speed_diff"
SPACE_HEADWAY = "Space_headway"
ID = "ID"
TIME = "Time"
COMPUTED_ACCELERATION = "Computed_acceleration"

# PARAMETERS
ALPHA = "alpha"
BETA = "beta"
GAMMA = "gamma"
LAMBDA = "lambda"
SIGMA = "sigma"

# STATE
ACC = "acc"
DEC = "dec"

# OTHER
DATA = "data"
LINSPACE = "linspace"
MEAN = "mean"
MEAN_ACC = "mean+"
MEAN_DEC = "mean-"
BETA_VARIABLE_CORRESPONDANCE = {
    ALPHA : {SPEED, SPACE_HEADWAY},
    BETA : {SPEED},
    GAMMA : {SPACE_HEADWAY},
    LAMBDA : {SPEED_DIFF},
    SIGMA : {}
}

class Plotter:
    def __init__(self, data: pd.DataFrame, parameters: dict):
        self.data=data
        self.parameters={k.replace("lamda", "lambda") : v for k, v in parameters.items()} #Repairing the typo in lambda

    def compute_acceleration(self, Speed, Space_headway, Speed_diff, betas):
        sensitivity_acc = betas["alpha_acc"]*((Speed+np.exp(-50))**betas["beta_acc"])/(Space_headway**betas["gamma_acc"])
        sensitivity_dec = betas["alpha_dec"]*((Speed+np.exp(-50))**betas["beta_dec"])/(Space_headway**betas["gamma_dec"])
        stimulus_acc = np.abs(Speed_diff + np.exp(-50))**betas["lambda_acc"]
        stimulus_dec = np.abs(Speed_diff + np.exp(-50))**betas["lambda_dec"]

        acc = sensitivity_acc * stimulus_acc
        dec = sensitivity_dec * stimulus_dec

        return acc*(Speed_diff >= 0) + dec*(Speed_diff < 0)

    def generate_data(self, Speed, Space_headway, Speed_diff, state= None):
        vars = {
            SPEED: Speed,
            SPACE_HEADWAY: Space_headway,
            SPEED_DIFF: Speed_diff
        }
        mask = self.data[SPEED_DIFF] >= 0.0 if state == ACC else self.data[SPEED_DIFF] < 0.0
        data = self.data.loc[mask] if state in {ACC,DEC} else self.data

        for var in vars.keys():
            # If the variable is DATA, take the corresponding value in the data
            if vars[var] == DATA:
                vars[var] = data[var]

            # Elif the variable is LINSPACE, make it a linspace between the max and the min of the data
            elif vars[var] == LINSPACE:
                min = data[var].min()
                max = data[var].max()
                vars[var] = np.linspace(min, max, len(data))

            # Elif the variable is a tuple, make it a linspace between those values (included)
            elif type(vars[var]) == tuple:
                vars[var] = np.linspace(*vars[var], len(data))
            
            # Elif the variable is MEAN, make it constant with the mean of the variable
            elif vars[var] == MEAN:
                vars[var] = data[var].mean()

            # Elif the variable is a number, keep it a constant
            elif isinstance(vars[var], (int, float)):
                pass

            #Else raise exception
            else:
                raise ValueError(f"Expected a number, tuple or '{DATA}', '{LINSPACE}' or '{MEAN}', but got {type(var).__name__}")
        
        return vars
    
    def plot_estimate(self, beta, var, plot_scatter=True):
        # Compute datas with var variating and all else being equal (AEBE)
        data_AEBE_ACC = self.generate_data(**{v : LINSPACE if v == var else MEAN  for v in {SPEED, SPACE_HEADWAY, SPEED_DIFF}}, state=ACC)
        data_AEBE_DEC = self.generate_data(**{v : LINSPACE if v == var else MEAN  for v in {SPEED, SPACE_HEADWAY, SPEED_DIFF}}, state=DEC)
        data = self.generate_data(**{v : DATA  for v in {SPEED, SPACE_HEADWAY, SPEED_DIFF}})
        #Sort the data by data[var]:
        data = pd.DataFrame(data).sort_values(var)
        betas_1 = {k : 1 if beta in k else v for k,v in self.parameters.items()}
        betas_0 = {k : 0 if beta in k else v for k,v in self.parameters.items()}

        
        plt.plot(data_AEBE_ACC[var], self.compute_acceleration(**data_AEBE_ACC, betas=self.parameters), color="C4", label = "Estimated value", zorder=2)
        plt.plot(data_AEBE_DEC[var], self.compute_acceleration(**data_AEBE_DEC, betas=self.parameters), color="C4", zorder=2)
        if plot_scatter:
            plt.plot(data[var], self.compute_acceleration(**data, betas=self.parameters), "+", markersize=1, alpha=0.5, color="C0", zorder= -2)
            plt.plot(data[var], self.data[ACCELERATION]                                 , "+", markersize=1, alpha=0.5, color="C1", zorder= -3)
            plt.plot(data[var], self.compute_acceleration(**data, betas=betas_1)        , ".", markersize=0.5, alpha=0.5, color="C2", zorder= -1)
            plt.plot(data[var], self.compute_acceleration(**data, betas=betas_0)        , ".", markersize=0.5, alpha=0.5, color="C3", zorder= -1)
        else:
            plt.plot(data_AEBE_ACC[var], self.compute_acceleration(**data_AEBE_ACC, betas=betas_1), "--",   color="C2", label = f"$\\{beta}$ = 1", zorder=1)
            plt.plot(data_AEBE_ACC[var], self.compute_acceleration(**data_AEBE_ACC, betas=betas_0), ":",   color="C3", label = f"$\\{beta}$ = 0", zorder=1)
            plt.plot(data_AEBE_DEC[var], self.compute_acceleration(**data_AEBE_DEC, betas=betas_1), "--",   color="C2", zorder=1)
            plt.plot(data_AEBE_DEC[var], self.compute_acceleration(**data_AEBE_DEC, betas=betas_0), ":",   color="C3", zorder=1)

        plt.vlines(0, *plt.ylim(), color="black", linewidth=1, zorder=0)
        plt.hlines(0, *plt.xlim(), color="black", linewidth=1, zorder=0)
        plt.vlines(0, *plt.ylim(), color="black", linewidth=1, zorder=0) #To update the ylims
        plt.xlabel(var)
        plt.ylabel("Acceleration")

        if plot_scatter: #artists to be visible in legend
            plt.plot(0, 0, "+", color="C0", label = "Estimated value, real data")
            plt.plot(0, 0, "+", color="C1", label = "Real value, real data")
            plt.plot(0, 0, ".", color="C2", label = f"$\\{beta}$ = 1, real data")
            plt.plot(0, 0, ".", color="C3", label = f"$\\{beta}$ = 0, real data")

            plt.title(f"$\\{beta}$, {var}")
        else:
            plt.title(f"$\\{beta}$, {var}, all else being equal")
        plt.legend()

    def plot_beta(self, beta):
        n = len(BETA_VARIABLE_CORRESPONDANCE[beta])

        plt.figure(figsize=(4*2, 5*n), layout="tight")
        for i, var in enumerate(BETA_VARIABLE_CORRESPONDANCE[beta]):
            plt.subplot(n, 2, i*2+1)
            self.plot_estimate(beta, var, plot_scatter=True)
            plt.subplot(n, 2, i*2+2)
            self.plot_estimate(beta, var, plot_scatter=False)




                

        
        
