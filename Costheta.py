################################################################################
#      Script to Compute Distribution Costheta in Polarized Samples            #
# Author: W. David Buitrago Ceballos (wilson.david.buitrago.ceballos@cern.ch)  #
################################################################################

# Libraries to use

import uproot
import awkward as ak
import vector
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import argparse 
import os

try:
    vector.register_awkward()
    print("[INFO] Libraty 'vector' Used with 'awkward'.")
except Exception as e:
    print(f"[INFO] 'vector' was already used or problem to use it: {e}")

def load_events_gen(file_path):
    """Load nanoAOD file."""
    print(f"[INFO] Load TTree events from the file: {file_path}")
    if not os.path.exists(file_path):
        print(f"[ERROR] The file not exist: {file_path}")
        return None
    try:
        file = uproot.open(file_path)
        if "Events" not in file:
            print(f"[ERROR] The TTree 'Events' doen't exist in the file: {file_path}")
            keys_in_file = [key.split(';')[0] for key in file.keys()]
            print(f"[INFO] TTrees available on the file: {keys_in_file}")
            return None

        events = file["Events"].arrays([
            "GenPart_pdgId", "GenPart_genPartIdxMother",
            "GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass",
        ], entry_stop=None)
        print(f"[INFO] {len(events)} Evenets Loaded.")
        if len(events) == 0:
            print("[WARN] TTree 'Events' not contains events.")
        return events
    except Exception as e:
        print(f"[ERROR] Not able to open the file {file_path}: {e}")
        return None

def get_w_decay_quark_pairs_gen(events):
    """
    Extraction of pairs of quarks coming from a W boson at generation lvl
    Gave a 'vector.Momentum4D'.
    """
    if events is None or len(events) == 0:
        return []
        
    pdgId = events["GenPart_pdgId"]
    mothers = events["GenPart_genPartIdxMother"]
    
    all_quark_pairs_vectors = []

    print(f"[INFO] Processing events to find W -> qq'.")
    num_w_decay_to_quarks = 0
    for i in range(len(events)): 
        event_pdgId = pdgId[i]
        event_mothers = mothers[i]
        
        # Label of the W boson in the actual event
        # abs(pdgId) == 24 for W+ y W-
        w_indices_in_event = ak.local_index(event_pdgId)[abs(event_pdgId) == 24]

        for w_idx in w_indices_in_event:
            # Indexes for the quarks doughters coming from to the W (abs(pdgId) entre 1 y 6)
            daughter_indices = ak.local_index(event_pdgId)[
                (event_mothers == w_idx) & (abs(event_pdgId) >= 1) & (abs(event_pdgId) <= 6)
            ]
            
            if len(daughter_indices) == 2:
                num_w_decay_to_quarks += 1
                q1_data = {
                    "pt": events["GenPart_pt"][i][daughter_indices[0]],
                    "eta": events["GenPart_eta"][i][daughter_indices[0]],
                    "phi": events["GenPart_phi"][i][daughter_indices[0]],
                    "mass": events["GenPart_mass"][i][daughter_indices[0]],
                }
                q2_data = {
                    "pt": events["GenPart_pt"][i][daughter_indices[1]],
                    "eta": events["GenPart_eta"][i][daughter_indices[1]],
                    "phi": events["GenPart_phi"][i][daughter_indices[1]],
                    "mass": events["GenPart_mass"][i][daughter_indices[1]],
                }
                q1_vec = vector.obj(pt=q1_data["pt"], eta=q1_data["eta"], phi=q1_data["phi"], mass=q1_data["mass"])
                q2_vec = vector.obj(pt=q2_data["pt"], eta=q2_data["eta"], phi=q2_data["phi"], mass=q2_data["mass"])
                all_quark_pairs_vectors.append([q1_vec, q2_vec])
                
    print(f"[INFO] Found {len(all_quark_pairs_vectors)} pairs of quarks {num_w_decay_to_quarks} for W->qq'.")
    return all_quark_pairs_vectors

def calculate_cos_theta_star_gen(quark_pairs_vectors):
    """
    Computation cos(theta*) for the list of pair of quarks.
    theta* is the angle betweeen the quark direction (in the W rest frame) and the dirección of W (in the lab frame).
    """
    cos_theta_stars = []

    if not quark_pairs_vectors:
        print(f"[WARN] Quark pairs not loaded for computation of cos(theta*).")
        return np.array([])

    for q_pair in quark_pairs_vectors:
        q1 = q_pair[0] 
        q2 = q_pair[1]
        w_lab = q1 + q2 
        if w_lab.mass < 1e-3 or w_lab.E <= 1e-6: 
            continue 

        # Boost q1 in the W rest frame W,ab.beta3
        try:
            q1_in_w_rest = q1.boost_beta3(-w_lab.beta3)
        except Exception as e:
            continue


        # Axis z' in the helicity frame
        w_direction_lab_3vec = w_lab.to_beta3() 

        if w_direction_lab_3vec.mag < 1e-6: 
            continue 
        
        q1_direction_w_rest_3vec = q1_in_w_rest.to_beta3()
        
        if q1_direction_w_rest_3vec.mag < 1e-6: 
            continue

        cos_theta = q1_direction_w_rest_3vec.unit().dot(w_direction_lab_3vec.unit())
        cos_theta_stars.append(cos_theta)
        
    print(f"[INFO] Computed {len(cos_theta_stars)} values of cos(theta*).")
    return np.array(cos_theta_stars)

def angular_model_gen(cos_theta, f0, f_minus, norm_factor):
    """
    Angular model for the decay of the W in function of cos(theta*).
    dN/d(cos_theta*) = N' * [ (3/8)(1-cos_theta)^2 f_L + (3/8)(1+cos_theta)^2 f_R + (3/4)sin^2(theta) f_0 ]
    Where f_L es f_minus, f_R es f_plus, y f_0 es f0.
    f_plus (f_R) from the restriction f0 + f_minus + f_plus = 1.
    norm_factor Factor form the global scale for adjust the normalize histogram.
    """
    f_plus = 1.0 - f0 - f_minus
    
    f0_eff = np.clip(f0, 0, 1)
    f_minus_eff = np.clip(f_minus, 0, 1 - f0_eff) 
    f_plus_eff = np.clip(1.0 - f0_eff - f_minus_eff, 0, 1)

    sum_f = f0_eff + f_minus_eff + f_plus_eff
    if not np.isclose(sum_f, 1.0) and sum_f > 1e-6:
        f0_eff /= sum_f
        f_minus_eff /= sum_f
        f_plus_eff /= sum_f
    elif sum_f <= 1e-6:
        f0_eff, f_minus_eff, f_plus_eff = 1/3, 1/3, 1/3


    # Terms of the Angular Diastribution:
    # (Using the convention where f_minus = f_L, f_plus = f_R)
    # sin^2(theta) = 1 - cos^2(theta)
    term_f0 = (3.0/4.0) * (1.0 - cos_theta**2) * f0_eff       # Para f0 (longitudinal)
    term_f_minus = (3.0/8.0) * (1.0 - cos_theta)**2 * f_minus_eff # Para f_L (left-handed)
    term_f_plus = (3.0/8.0) * (1.0 + cos_theta)**2 * f_plus_eff   # Para f_R (right-handed)
    
    dist = term_f0 + term_f_minus + term_f_plus
    return dist * norm_factor

def fit_and_plot_gen(cos_theta_stars, plot_label, ax, output_file=None):
    """
   Histograms, Fitting and plotting.
    """
    if len(cos_theta_stars) == 0:
        print(f"[WARN] No data for cos(theta*) not possible to plot/fit for {plot_label}.")
        ax.text(0.5, 0.5, "No data for $\cos\theta^*$", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f"Distribution $\cos\theta^*$ ({plot_label})")
        return None 

    # Normalize Histogram
    hist_values, bin_edges = np.histogram(cos_theta_stars, bins=50, range=(-1, 1), density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0] 

    # Fitting
    try:
        # Initial Guess: [f0, f_minus, factor_de_normalización_histograma]
        # If density=True, norm_factor should be close to 1.
        # If W is dominant - left-handed (SM V-A), f_minus can be big.
        initial_guess = [0.3, 0.5, 1.0] 
        # Limits: Fractions between 0 y 1. norm_factor positive.
        # f0 + f_minus <= 1
        bounds_fit = ([0, 0, 0.01], [1, 1, np.inf]) 
        
        popt, pcov = curve_fit(angular_model_gen, bin_centers, hist_values, 
                               p0=initial_guess, bounds=bounds_fit, maxfev=10000,
                               sigma=np.sqrt(hist_values / (len(cos_theta_stars) * bin_width)) if np.all(hist_values >0) else None
                               ) 
                               

        f0_fit, f_minus_fit, norm_factor_fit = popt
        f_plus_fit = 1.0 - f0_fit - f_minus_fit

        f_sum = f0_fit + f_minus_fit + f_plus_fit
        if not np.isclose(f_sum, 1.0) and f_sum > 0:
            f0_fit /= f_sum
            f_minus_fit /= f_sum
            f_plus_fit /= f_sum
        
        f0_fit = np.clip(f0_fit, 0, 1)
        f_minus_fit = np.clip(f_minus_fit, 0, 1)
        f_plus_fit = np.clip(f_plus_fit, 0, 1)


        # Plotting
        ax.hist(cos_theta_stars, bins=50, range=(-1, 1), density=True, 
                histtype='step', label=f"Datos ({plot_label})", color='blue', linewidth=1.5)
        
        x_fit_vals = np.linspace(-1, 1, 200)
        y_fit_vals = angular_model_gen(x_fit_vals, f0_fit, f_minus_fit, norm_factor_fit)
        ax.plot(x_fit_vals, y_fit_vals, label=f"Ajuste Modelo Angular", linestyle='--', color='red', linewidth=2)

        formula_text = (
            r"$\frac{dN}{d\cos\theta^*} = N' \cdot \left[ \frac{3}{4}(1-\cos^2\theta^*)f_0 \right.$" + "\n" +
            r"$\left. + \frac{3}{8}(1-\cos\theta^*)^2 f_{-} + \frac{3}{8}(1+\cos\theta^*)^2 f_{+} \right]$"
        )
        fit_params_text = (
            fr"$f_0$ (longitudinal) = {f0_fit:.3f}" + "\n" +
            fr"$f_-$ (left) = {f_minus_fit:.3f}" + "\n" +
            fr"$f_+$ (right) = {f_plus_fit:.3f}" + "\n" +
            fr"$N'$ (norm) = {norm_factor_fit:.3f}"
        )
        
        # Add chi2/ndof if it is possible
        residuals = hist_values - angular_model_gen(bin_centers, *popt)
        chi2 = np.sum((residuals**2) / hist_values[hist_values > 0]) 
        ndof = len(bin_centers[hist_values > 0]) - len(popt)
        chi2_text = ""
        if ndof > 0:
            chi2_text = fr"$\chi^2$/ndof = {chi2/ndof:.2f}"


        ax.text(0.03, 0.97, formula_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', fc='aliceblue', alpha=0.8))
        ax.text(0.65, 0.35, fit_params_text + "\n" + chi2_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.8))

        fit_results = {'f0': f0_fit, 'f_minus': f_minus_fit, 'f_plus': f_plus_fit, 'norm': norm_factor_fit, 'chi2_ndof': chi2/ndof if ndof > 0 else -1}

    except RuntimeError as e:
        print(f"[ERROR] Not able to do the fitting for {plot_label}: {e}")
        ax.hist(cos_theta_stars, bins=50, range=(-1, 1), density=True, 
                histtype='step', label=f"Datos ({plot_label}) - Sin ajuste", color='blue')
        fit_results = None
    except ValueError as e:
        print(f"[ERROR] Valur problem during Fitting {plot_label} (error should be given by bounds, p0 or emty/invalid data): {e}")
        ax.hist(cos_theta_stars, bins=50, range=(-1, 1), density=True, 
                histtype='step', label=f"Datos ({plot_label}) - Sin ajuste", color='blue')
        fit_results = None

    ax.set_title(f"Distribución de $\cos\theta^*$ (Nivel Generador - {plot_label})", fontsize=14)
    ax.set_xlabel(r"$\cos\theta^*$", fontsize=12)
    ax.set_ylabel("Densidad Normalizada", fontsize=12)
    ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=0) 

    return fit_results


def main():
    parser = argparse.ArgumentParser(description="Analyze the W polarization computing cos(theta*) "
                                                 "from nanoAOD file, make the fitting and the plot.")
    parser.add_argument("input_file", type=str, help="Path nanoAOD file.")
    parser.add_argument("--label", type=str, default="Muestra", help="Label for the plot (ej. Sample name).")
    parser.add_argument("--output_plot", type=str, default=None, help="optional path for the output - plot ('plot.png').")
    parser.add_argument("--max_events", type=int, default=None, help="Maxim number of events to process in the nanoAOD file (opcional).")


    args = parser.parse_args()

    def load_events_gen_modified(file_path, entry_stop=None):
        """Load de branches from the nanoAOD."""
        print(f"[INFO] Loading events from: {file_path}" + (f" (until {entry_stop} events)" if entry_stop else ""))
        if not os.path.exists(file_path):
            print(f"[ERROR] File doesn't exist: {file_path}")
            return None
        try:
            file = uproot.open(file_path)
            if "Events" not in file:
                print(f"[ERROR] TTree 'Events' was not founded in the file: {file_path}")
                keys_in_file = [key.split(';')[0] for key in file.keys()]
                print(f"[INFO] TTrees available in the file: {keys_in_file}")
                return None

            events = file["Events"].arrays([
                "GenPart_pdgId", "GenPart_genPartIdxMother",
                "GenPart_pt", "GenPart_eta", "GenPart_phi", "GenPart_mass",
            ], entry_stop=entry_stop)
            print(f"[INFO] {len(events)} Events Loades Succesfully.")
            if len(events) == 0 and entry_stop != 0:
                 print("[WARN] The file dopnt have events to show.")
            return events
        except Exception as e:
            print(f"[ERROR] Not possible to read the file {file_path}: {e}")
            return None

    # 1. Load Events
    events_data = load_events_gen_modified(args.input_file, args.max_events)
    if events_data is None or len(events_data) == 0:
        print(f"[EXIT] Not able to charge events: {args.input_file}")
        return

    # 2. Extraction of pairs of W
    quark_pairs = get_w_decay_quark_pairs_gen(events_data)
    if not quark_pairs:
        print(f"[WARN] Decays not founded for W -> qq' in {args.input_file}")
       
    # 3. Computing cos(theta*)
    cos_theta_values = calculate_cos_theta_star_gen(quark_pairs)
    
    fig, ax = plt.subplots(figsize=(10, 7)) 

    # 4. Fitting and Plotting
    print(f"\n[INFO] Doing Fitting and plotting: {args.label}")
    fit_results = fit_and_plot_gen(cos_theta_values, args.label, ax)

    if fit_results:
        print("\n[INFO] Fitting Results:")
        print(f"  f0 (longitudinal): {fit_results['f0']:.4f}")
        print(f"  f- (left):        {fit_results['f_minus']:.4f}")
        print(f"  f+ (right):       {fit_results['f_plus']:.4f}")
        print(f"  Norm. Factor:     {fit_results['norm']:.4f}")
        if fit_results['chi2_ndof'] != -1:
             print(f"  Chi^2/ndof:       {fit_results['chi2_ndof']:.2f}")
    else:
        print("[INFO] Not able to obtain valid values for the fitting.")

  
    plt.tight_layout(rect=[0, 0.05, 1, 0.96]) 

    if args.output_plot:
        try:
            plt.savefig(args.output_plot)
            print(f"\n[INFO] Plot saved in: {args.output_plot}")
        except Exception as e:
            print(f"[ERROR] Not able to save file in {args.output_plot}: {e}")
    
    plt.show()


if __name__ == "__main__":
    main()