import pandas as pd # for data frame processing
import numpy as np # for some statistical procedures
from scipy.optimize import minimize # optimisation routine for parameter estimation
from scipy.stats import t # normal distribution density function
import numdifftools as nd # we use this to calculate t-ratios and p-values
import csv # we need this to store our parameters as csv


def optimizer(betas_start, SLL, model_name, verbose=2, pbar=None):
    # This will give us the initial loglikelihood value as an output
    def callback1(betas):
        print("Current log likelihood:", -SLL(betas))

    # This function will allow as to store parameter estimates during iterations
    # Initialise list to store parameter values
    parameter_values = [np.array(list(betas_start.values()))]
    # Then define the function
    def callback2(betas):
        parameter_values.append(betas)
        column_names = list(betas_start.keys())
        with open(f'outputs/{model_name}_iterations.csv','w',newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)
            writer.writerows(parameter_values)

    # Now let's combine the two callback functions
    def combined_callback(betas):
        if verbose>1: 
            callback1(betas)
        callback2(betas)

    if verbose > 0:        
        print("Initial log likelihood:", -SLL(np.array(list(betas_start.values()))))

    # Choose optimisation routine (preferably BFGS)
    optimiser = 'BFGS' # BFGS or L-BFGS-B or nelder-mead

    
    # -If a tqdm progressbar is provided, start with a low gtol and if it does not work increase it :
    if pbar is not None:
        gtol=1e-3
        pbar.set_postfix({"gtol":gtol})

        result = minimize(SLL, np.array(list(betas_start.values())), method=optimiser,callback=combined_callback, 
                        options={'disp':False, "gtol":gtol}) # ,bounds=bounds1
            
        while not result["success"]:
            gtol += 1e-3
            pbar.set_postfix({"gtol":gtol}) 
            result = minimize(SLL, np.array(list(betas_start.values())), method=optimiser,callback=combined_callback, 
                        options={'disp':False, "gtol":gtol}) # ,bounds=bounds1
        
        message = pbar.desc
        if message ==  "":
            message = "gtol :"
        pbar.set_description(message + f" {model_name}:{gtol:.3f}")
    #Else just compute it once
    else:
        result = minimize(SLL, np.array(list(betas_start.values())), method=optimiser,callback=combined_callback, 
                        options={'disp':False}) # ,bounds=bounds1
         


    #args = (parameter_values,)
    if verbose > 0:
        print("Final log likelihood:", -result.fun)

    return result

def calculate_results(result, betas_start, SLL, LL, Nobs, verbose=2):
    # Vector of parameter estimates
    parameters = result['x'] 

    # Calculate hessian
    if verbose > 0:
        print("Calculating Hessian, please wait (this may take a while...)")
    Hess = nd.Hessian(SLL)
    hessian = Hess(parameters)
    inv_hessian = np.linalg.inv(hessian)

    # Parameter statistics
    dof = Nobs - len(betas_start) - 1
    se = np.sqrt(np.diag(inv_hessian)) # Standard errors
    tratio = parameters/se # t-ratios
    p_value = (1-t.cdf(np.abs(tratio),dof)) * 2 # p-values


    # --- Sandwich estimator --- #

    # The sandwich estimator provides the "robust" s.e., t-ratios and p-values.
    # These should be preferred over the classical parameter statistics.

    # We first need the gradients at the solution point
    Grad = nd.Gradient(LL)
    gradient = Grad(parameters)

    # Then we need to calculate the B matrix
    B = np.array([])
    for r in range(gradient.shape[0]):
        Bm = np.zeros([len(betas_start),len(betas_start)])
        gradient0 = gradient[r,:]
        for i in range(len(gradient0)):
                for j in range(len(gradient0)):
                    element = gradient0[i]*gradient0[j]
                    Bm[i][j] = element
        if B.size==0:
                        B = Bm
        else:
                        B = B+Bm

    # Finally we "sandwich" the B matrix between the inverese hessian matrices
    BHHH = (inv_hessian)@(B)@(inv_hessian)

    if verbose > 0:
        print("... Done!!")

    # Now it is time to calculate some "robust" parameter statistics
    rob_se = np.sqrt(np.diag(BHHH)) # robust standard error
    rob_tratio = parameters/rob_se # robust t-ratio
    rob_p_value = (1-t.cdf(np.abs(rob_tratio),dof)) * 2 # robust p-value

    arrays = np.column_stack((np.array(list(betas_start.keys())),parameters,se,tratio,p_value,rob_se,rob_tratio,rob_p_value))
    results = pd.DataFrame(arrays, columns = ['Parameter','Estimate','s.e.','t-ratio0','p-value',
                                            'Rob s.e.','Rob t-ratio0','Rob p-value'])

    results[['Estimate','s.e.','t-ratio0','p-value','Rob s.e.','Rob t-ratio0','Rob p-value']] = (
    results[['Estimate','s.e.','t-ratio0','p-value','Rob s.e.','Rob t-ratio0','Rob p-value']].apply(pd.to_numeric,errors='coerce'))
    numeric_cols = results.select_dtypes(include='number').columns
    results[numeric_cols] = results[numeric_cols].round(3)
    return results

def calculate_goodness_of_fit(result, betas_start, SLL, Nobs, **_):
    gof = {}
    gof["rho squared"] = 1 - ((-result.fun)/(-SLL(np.zeros(len(betas_start)))))[0]
    gof["adjusted rho squared"] = 1 - (((-result.fun)-len(betas_start))/(-SLL(np.zeros(len(betas_start)))))[0]

    gof["AIC"] = 2*len(betas_start) - 2*(-result.fun)
    gof["BIC"] = len(betas_start)*np.log(Nobs) - 2*(-result.fun)

    gof["Log likelihood at zeros:"] = -SLL(np.zeros(len(betas_start)))[0]
    gof["Initial log likelihood:"] = -SLL(np.array(list(betas_start.values())))[0]
    gof["Final log likelihood:"] = -result.fun

    return gof