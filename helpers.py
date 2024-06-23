import pandas as pd # for data frame processing
import numpy as np # for some statistical procedures
import pickle


def save(obj, name, verbose=1):
    with open(f"pickles/{name}.pickle", 'wb') as f:
        pickle.dump(obj, f)
    if verbose > 0:
        print(f"Saved to 'pickles/{name}.pickle'")

def load(name, verbose=1):
    if verbose > 0:
        print(f"Loading 'pickles/{name}.pickle'")
    with open(f"pickles/{name}.pickle", 'rb') as f:
        return pickle.load(f)

def show_significance(df:  pd.DataFrame, level=0.05):
    return df.style.apply(
          lambda s, props: np.full(len(s), np.where(s["Rob p-value"] > level, props, '')),
          props='color:darkred',
          axis=1).hide(level=0)
    
def style_ttest(serie:  pd.Series, level=1.96):
    return pd.DataFrame(serie).style.apply(
          lambda s, props: np.full(len(s), np.where(s.abs() > level, props, '')),
          props='color:darkgreen;font-weight:bold;',
          axis=1)

def style_gof(df: pd.DataFrame, index_is_time = True):
    vmin =df.iloc[:,4:].min(axis=None)
    if index_is_time:
        dfstyle = (
            df.reset_index(names="Reaction time").style.hide(level=0)
            .format("{:n} s", subset=["Reaction time"])
            )
    else:
        dfstyle = df.style
    return (
        dfstyle
        .highlight_max(axis=0, subset=["rho squared", "adjusted rho squared"], props="background: darkblue; color: white;")
        .highlight_min(axis=0, subset=["AIC", "BIC"], props="background: darkblue; color: white;")
        .background_gradient(axis=None, cmap="Blues", subset = df.columns[4:], vmin=vmin)
        .format(precision=2, subset=df.columns)
        .set_caption("Goodness of fit measures")
    )

def show_latent_models(results):
    names = list(results.keys())
    parameters = [result.set_index("Parameter")["Estimate"].to_dict() for result in results.values()]



    def utilities(parameters):
        out = "*Utilities*"
        out += f" $$\\begin{{gathered}} V^{{acc}} = {parameters['bc_acc']} + {parameters['b_speed_c1']} * \\Delta V_n(t-\\tau)"
        if 'b_space_headway_c1' in parameters:
            out += f" + {parameters['b_space_headway_c1']} * \\Delta X_n(t-\\tau)"
        out += f"\\\\ V^{{dec}} = {parameters['bc_dec']} + {parameters['b_speed_c2']} * \\Delta V_n(t-\\tau)"
        if 'b_space_headway_c2' in parameters:
            out += f" + {parameters['b_space_headway_c2']} * \\Delta X_n(t-\\tau)"
        out += "\\\\ V^{dn} = 0"
        out += "\\end{gathered}$$ "

        return out     

    def GM(parameters):
        return ("*Accelerating*"
                f" $$\\alpha_n(t) = e^{{{parameters['alpha_acc']}}} \\cdot \\frac{{V_n(t)^{{{parameters['beta_acc']}}}}}{{\Delta X_n(t)^{{{parameters['gamma_acc']}}}}} \\cdot \\left\\|\\Delta V_n\\left(t-\\tau_n\\right)\\right\\|^{{{parameters['lamda_acc_p']}}}$$"
                f" $$\\alpha_n(t) = {np.exp(parameters['alpha_acc']):.3f} \\cdot \\frac{{V_n(t)^{{{parameters['beta_acc']}}}}}{{\Delta X_n(t)^{{{parameters['gamma_acc']}}}}} \\cdot \\left\\|\\Delta V_n\\left(t-\\tau_n\\right)\\right\\|^{{{parameters['lamda_acc_p']}}}$$"
                "*Decelerating*"
                f" $$\\alpha_n(t) = e^{{{parameters['alpha_dec']}}} \\cdot \\frac{{V_n(t)^{{{parameters['beta_dec']}}}}}{{\Delta X_n(t)^{{{parameters['gamma_dec']}}}}} \\cdot \\left\\|\\Delta V_n\\left(t-\\tau_n\\right)\\right\\|^{{{parameters['lamda_dec_n']}}}$$"
                f" $$\\alpha_n(t) = {np.exp(parameters['alpha_dec']):.3f} \\cdot \\frac{{V_n(t)^{{{parameters['beta_dec']}}}}}{{\Delta X_n(t)^{{{parameters['gamma_dec']}}}}} \\cdot \\left\\|\\Delta V_n\\left(t-\\tau_n\\right)\\right\\|^{{{parameters['lamda_dec_n']}}}$$"
        )

    def other_variables(parameters):
        out = ("*Other variables*<br>"
               f"$\\mu_{{dn}} = {parameters['dn']:.2f}$<br>")
        for var in ["sigma_{acc}" ,"sigma_{dec}" ,"sigma_{dn}"]:
            out += f"$\\{var} = {parameters[var.replace("{", "").replace("}", "")]:.2f}$<br>"
        return out


    markdown = ""

    # Title
    markdown += "|" + "|".join(names) + "|" +"\n| ---- | ---- |\n"

    # Utilities
    markdown += "|" + "|".join([utilities(p) for p in parameters]) + "|" + "\n"

    # GM model
    markdown += "|" + "|".join([GM(p) for p in parameters]) + "|" + "\n"

    # Other variables
    markdown += "|" + "|".join([other_variables(p) for p in parameters]) + "|" + "\n"

    return(markdown)
