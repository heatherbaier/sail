import torch
import typer
import numpy as np
import pandas as pd
from haversine import haversine, Unit
from ..utils.utils import *
from ..data.instance_index import *
import torch.nn.functional as f
import matplotlib.pyplot as plt
import os

import tqdm

import warnings
warnings.filterwarnings('ignore')


class Simba:
    def __init__(
        self,
        model,
        device = "cuda",
        baseline_image=None,
        baseline_coords=None,
        label = None,
        additional_model_inputs = None,
        coord_names=("lat", "lon"),
        distance_fn=None,
        perturbation_scheme="small_local",
        ckpt_dir = None
    ):
        self.model = model.to(device)
        self.device = device
        self.baseline_image = baseline_image
        self.baseline_coords = baseline_coords
        self.label = label,
        self.coord_names = coord_names
        self.distance_fn = distance_fn or (lambda c1, c2: haversine(c1, c2, unit=Unit.KILOMETERS))
        self.perturbation_scheme = perturbation_scheme
        self.additional_model_inputs = additional_model_inputs
        self.ckpt_dir = ckpt_dir

        if self.baseline_image is not None:
            _, _, self.baseline_pred = self.get_gradients_wrt_inputs(self.baseline_image, torch.tensor(self.baseline_coords, dtype=torch.float32).unsqueeze(0))
        # print("BASELINE PREDICTION: ", self.baseline_pred)

    def perturb_coordinates(self, coords, distance_km):
        """
        Generate perturbed coordinates at a given radial distance from coords.
        Uses small angular offsets based on the distance in km.
        """
        # print("COORDS: ", coords)
        lat, lon = coords
        offsets = [
            (lat + distance_km / 111.32, lon),  # north
            (lat - distance_km / 111.32, lon),  # south
            (lat, lon + distance_km / (111.32 * np.cos(np.radians(lat)))),  # east
            (lat, lon - distance_km / (111.32 * np.cos(np.radians(lat)))),  # west
        ]
        
        return offsets


    def explain_instance(self, distances_km):

        df = self.run_instance_analysis(distances_km = distances_km)
        self.export_results(df)
        
        # Get, save and print stats    
        metrics = self.summarize_variogram_from_df(df)
        self.plot_variogram_summary()
        res = self.simba_variogram_with_null(distances_km)
        self.print_null_variogram_summary(res)
        self.export_results(df)
        typer.echo(f"Saved instance sensitivity metrics!")


    def explain_global_from_list(
        self,
        instances,
        distances_km,
        agg="mean",                # "mean" | "median"
        save_csv="artifacts/global_variogram.csv",
        save_per_instance=False,   # also save each instance’s DF if True
        max_instances=None,
    ):
        """
        Global SIMBA summary over many instances.
    
        Parameters
        ----------
        instances : list of dict
            Each dict must minimally contain:
              {"image_path": <str>, "coord": (lat, lon)}
            If your class keeps label/extra inputs internally, you don't need them here.
        distances_km : iterable of float
            Radii for perturbations (same as single-instance).
        agg : str
            Aggregator for global curve: "mean" or "median".
        save_csv : str | None
            Where to write the global table (per-distance aggregates).
        save_per_instance : bool
            If True, also export each instance df via your existing export_results(df).
        max_instances : int | None
            If set, subsample the first N instances for speed.
    
        Returns
        -------
        dict with:
          global_df, global_curve (per-distance vector),
          global_metrics (nugget/slope/sill/eff_range),
          per_instance_metrics (list), per_instance_null (list)
        """
        import pandas as pd
        import numpy as np
        from copy import deepcopy
    
        # Storage
        all_rows = []
        per_instance_metrics = []
        per_instance_null = []
    
        # Subsample if needed
        # iterable = instances[:max_instances] if max_instances else instances

        iterable = instances
    
        for idx, inst in tqdm.tqdm(enumerate(iterable), desc="Explaining global..."):


            # print(inst.keys())

            # agda

            self.baseline_image = inst["image"]
            self.baseline_coords = inst["coords"]
            self.label = inst["label"]
            
            # --- set the instance on the fly ---
            self.image_path = inst["image_name"]
            self.coord = tuple(inst["coords"])
            _, _, self.baseline_pred = self.get_gradients_wrt_inputs(self.baseline_image, torch.tensor(self.baseline_coords, dtype=torch.float32).unsqueeze(0))
    
            # Run your existing single-instance pieces
            df_i = self.run_instance_analysis(distances_km=distances_km)   # must include distance_km & gamma
            df_i = df_i.copy()
            df_i["instance_id"] = idx
            df_i["image_path"] = self.image_path
            df_i["coord_lat"] = self.coord[0]
            df_i["coord_lon"] = self.coord[1]
    
            if save_per_instance:
                self.export_results(df_i)
    
            # Summaries (observed variogram stats from this instance)
            m_i = self.summarize_variogram_from_df(df_i)   # expects distance_km + gamma
            per_instance_metrics.append({"instance_id": idx, **m_i})
    
            # Null baseline for this instance
            res_i = self.simba_variogram_with_null(distances_km=distances_km)
            per_instance_null.append({"instance_id": idx, **res_i})
    
            all_rows.append(df_i)

            if max_instances is not None:
                if idx > max_instances:
                    break
            
    
        # --- Concatenate all instances ---
        if not all_rows:
            raise ValueError("No instances processed.")
        big = pd.concat(all_rows, ignore_index=True)
    
        # --- Aggregate to a global curve (per distance) ---
        # keep both mean and std so you can plot error bars
        agg_fun = "mean" if agg == "mean" else "median"
        # ensure we only aggregate numeric columns safely
        global_agg = big.groupby("distance_km").agg(
            gamma_mean=("gamma", "mean"),
            gamma_std =("gamma", "std"),
            gamma_median=("gamma", "median"),
            coord_sens_mean=("coord_sens", "mean") if "coord_sens" in big.columns else ("gamma", "mean"),
            coord_sens_std =("coord_sens", "std")  if "coord_sens" in big.columns else ("gamma", "std")
        ).reset_index()
    
        # Choose which series is the "global curve" for metrics
        ycol = "gamma_mean" if agg_fun == "mean" else "gamma_median"
        df_for_metrics = global_agg.rename(columns={ycol: "gamma"})[["distance_km", "gamma"]].dropna()
    
        # Use your existing summarizer on the aggregated curve
        global_metrics = self.summarize_variogram_from_df(df_for_metrics)
    
        # Optional: save global table
        if save_csv:
            filepath1 = os.path.join(self.ckpt_dir, f"sensitivity_index_global.csv")
            filepath2 = os.path.join(self.ckpt_dir, f"sensitivity_index_globalbig.csv")
            global_agg.to_csv(filepath1, index=False)
            big.to_csv(filepath2, index=False)


        # self.metrics = {
        #                 "nugget": float(nugget),
        #                 "slope0": float(slope0),
        #                 "sill": sill,
        #                 "eff_range_km": float(eff_range) if not np.isnan(eff_range) else np.nan,
        #                 "distances": hs,
        #                 "gammas": gs
        #             }
    
        # return self.metrics

        self.metrics = {
                        "nugget": global_metrics['nugget'],
                        "slope0": global_metrics['slope0'],
                        "sill": global_metrics['sill'],
                        "eff_range_km": global_metrics['eff_range_km'] if global_metrics['eff_range_km']==global_metrics['eff_range_km'] else float('nan'),
                        "distances": distances_km,
                        "gammas": global_agg["gamma_mean"]
                    }

        self.plot_variogram_summary(g = True)
        
        # Optional: quick plot using your existing plotter if it uses internal state
        # If your plotter pulls from self, you can adapt it; otherwise
        # write a small plotting helper that accepts (dist, mean, std).
    
        # --- Compact terminal + file summary ---
        summary_lines = []
        summary_lines.append("=" * 72)
        summary_lines.append(" SIMBA Global Variogram Summary")
        summary_lines.append("=" * 72)
        summary_lines.append(f" Instances: {len(iterable)}")
        summary_lines.append(f" Aggregation: {agg_fun.upper()} per distance over instances")
        summary_lines.append(f" Nugget (global): {global_metrics['nugget']}")
        summary_lines.append(f" Slope@0 (global): {global_metrics['slope0']}")
        summary_lines.append(f" Sill   (global): {global_metrics['sill']}")
        eff_range = global_metrics['eff_range_km']
        summary_lines.append(f" Eff. range ~95% (global): {eff_range if eff_range==eff_range else float('nan')} km")

        summary_lines.append("-" * 72)
        summary_lines.append(" Distance  γ_mean   (±1 std)   |  γ_median")
        
        for _, r in global_agg.iterrows():
            d = r["distance_km"]; gm = r["gamma_mean"]; gs = r["gamma_std"]; med = r["gamma_median"]
            summary_lines.append(f"{d}  {gm} (±{(gs if not np.isnan(gs) else 0)}) | {med}")

        
        summary_lines.append("=" * 72)
        
        # Print to terminal
        for line in summary_lines:
            print(line)

        # Write to file
        with open(os.path.join(self.ckpt_dir, "simba_global_variogram_summary.txt"), "w") as f:
            for line in summary_lines:
                f.write(line + "\n")

    
        return {
            "global_df": global_agg,
            "global_curve": df_for_metrics,
            "global_metrics": global_metrics,
            "per_instance_metrics": per_instance_metrics,
            "per_instance_null": per_instance_null
        }


        

        

    def get_gradients_wrt_inputs(self, image, coords):
        """
        Compute gradients of prediction wrt image pixels and coordinate inputs.
        """
        self.model.zero_grad()
        self.model.eval()

        # import random

        # print("COORDS HERE: ", coords)
        
        image = image.to(self.device).unsqueeze(0).detach().requires_grad_(True) 
        coords = coords.to(self.device).detach().requires_grad_(True)

        # coords = torch.tensor([[ random.random(), random.random()]])
        # coords = coords.to(self.device).detach().requires_grad_(True)

        # print("COORDS HERE: ", coords)


        inputs = [image, coords]

        if self.additional_model_inputs is not None:
            for ami in self.additional_model_inputs:
                ami = ami.to(self.device).unsqueeze(0).detach().requires_grad_(True)
                inputs.append(ami)
    
        
        with torch.enable_grad():

            y = (self.model.net if hasattr(self.model, "net") else self.model)(*inputs)
            y_scalar = y.reshape(-1).mean()

            # print("Prediction: ", y_scalar)

            # adga
            
            # y = self.model(*inputs)   # expect shape [1] or scalar
            # pick a scalar to differentiate; for regression, mean() is safe
            # y_scalar = y.reshape(-1).mean()

        
    
        # Gradients wrt coords and image
        g_image, g_coords = torch.autograd.grad(
            y_scalar, (image, coords),
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )

        # print("G coords: ", g_coords, "\n")
        # print("G image: ", g_image.shape)
        # print("Y: ", y)

        return g_image, g_coords, y_scalar

        # return image_grad, g_coords

    def compute_sensitivity(self, image, coords, normalize_by_distance=False, distance_km=None):
        """
        Calculate average gradient magnitudes for coords and image.
        """
        image_grad, coord_grad, y_scalar = self.get_gradients_wrt_inputs(image, coords)

        # print("CG: ", coord_grad, "\n")
        # print("IMG: ", image_grad)
        
        image_sens = image_grad.abs().mean().item()
        coord_sens = coord_grad.abs().mean().item()

        # print(y_scalar, self.baseline_pred)


        gamma = 0.5 * np.mean((y_scalar.detach().cpu().numpy() - self.baseline_pred.detach().cpu().numpy())**2)

        # print("GAMMA: ", gamma)

        # print("image sens: ", image_sens)
        # print("coord sens: ", coord_sens)

        if normalize_by_distance and distance_km:
            coord_sens /= distance_km

        return {
            "image_sens": image_sens,
            "coord_sens": coord_sens,
            "normed_coord_sens": coord_sens if normalize_by_distance else None,
            "gamma": gamma,
            "y_pred": y_scalar.item() 
        }

    def run_instance_analysis(self, distances_km):
        """
        Run perturbations for one baseline instance.
        """
        results = []
        for dist in distances_km:
            perturbed_coords = self.perturb_coordinates(self.baseline_coords, dist)
            # print("dist: ", dist)#, "     |      PC: ", perturbed_coords)
            for pcoords in perturbed_coords:
                coords_tensor = torch.tensor(pcoords, dtype=torch.float32).unsqueeze(0)
                sens = self.compute_sensitivity(self.baseline_image, coords_tensor,
                                                normalize_by_distance=True,
                                                distance_km=dist)
                sens["distance_km"] = dist
                results.append(sens)

            

        df = pd.DataFrame(results)
        # df = df.groupby(df["distance_km"]).mean().reset_index()
        self.df = df
        return df

    def run_global_analysis(self, sample_images, sample_coords, distances_km):
        """
        Aggregate sensitivity across many samples.
        """
        all_results = []
        for img, coords in zip(sample_images, sample_coords):
            self.baseline_image = imgs
            self.baseline_coords = coords
            df = self.run_instance_analysis(distances_km)
            all_results.append(df)
        return pd.concat(all_results, ignore_index=True)

    def plot_sensitivity_map(self, results_df):
        """
        Placeholder for mapping sensitivity (folium/geopandas/matplotlib).
        """
        raise NotImplementedError("Mapping not yet implemented.")

    def export_results(self, results_df):
        """
        Save results to CSV.
        """
        if not os.path.exists(os.path.join(self.ckpt_dir, "sensitivity_results")):
            os.mkdir(os.path.join(self.ckpt_dir, "sensitivity_results"))
        filepath = os.path.join(self.ckpt_dir, "sensitivity_results", f"sensitivity_index_{self.baseline_coords}.csv")
        results_df.to_csv(filepath, index=False)

    def summarize_variogram_from_df(self, df: pd.DataFrame, gamma_col="gamma", h_col="distance_km",
                                eff_level=0.95):
        """
        df must have columns:
          - h_col: distances in km
          - gamma_col: semivariance γ(h)
    
        Returns dict with nugget, slope0, sill, eff_range_km (+ a few extras).
        """

        # df = self.df.groupby(self.df["distance_km"]).mean().reset_index()
        
        # 1) Clean & sort
        d = df[[h_col, gamma_col]].dropna().drop_duplicates().sort_values(h_col).to_numpy()
        hs = d[:, 0].astype(float)
        gs = d[:, 1].astype(float)
    
        if len(hs) < 1:
            raise ValueError("No data to summarize.")
        if len(hs) == 1:
            nugget = gs[0]
            return {
                "nugget": nugget,
                "slope0": np.nan,
                "sill": gs[0],
                "eff_range_km": np.nan,
                "distances": hs,
                "gammas": gs
            }
    
        # 2) Core summaries
        nugget = gs[0]
        slope0 = (gs[1] - gs[0]) / (hs[1] - hs[0])  # finite-difference near origin
        sill = float(np.max(gs))
    
        # 3) Effective range: first h where γ(h) reaches eff_level * sill, with interpolation
        if sill <= 0:
            eff_range = np.nan
        else:
            target = eff_level * sill
            # find first index where gamma >= target
            idx = np.argmax(gs >= target)  # returns 0 if first already >= target
            if gs[idx] < target:  # never reaches target
                eff_range = np.nan
            elif idx == 0:
                eff_range = hs[0]
            else:
                # linear interpolate between (idx-1, idx)
                h0, g0 = hs[idx-1], gs[idx-1]
                h1, g1 = hs[idx],   gs[idx]
                if g1 == g0:
                    eff_range = h1
                else:
                    t = (target - g0) / (g1 - g0)
                    eff_range = h0 + t * (h1 - h0)

        self.metrics = {
                        "nugget": float(nugget),
                        "slope0": float(slope0),
                        "sill": sill,
                        "eff_range_km": float(eff_range) if not np.isnan(eff_range) else np.nan,
                        "distances": hs,
                        "gammas": gs
                    }
    
        return self.metrics


    def plot_variogram_summary(self, title=None, g = False):

        metrics = self.metrics

        temp = self.df.groupby(self.df["distance_km"]).mean().reset_index()
        print(temp)
        hs = temp["distance_km"]; gs = temp["gamma"]

        

        print(hs)
        print("~~~~~~~~~~~~~~~~~~~")
        print(gs)
        
        # hs = metrics["distances"]; gs = metrics["gammas"]
        
        plt.clf()
        plt.figure(figsize=(5,3.5))
        plt.plot(hs, gs, marker='o')
        plt.xlabel("Perturbation distance h (km)")
        plt.ylabel("Semivariance γ(h)")
        if title: plt.title(title)
    
        # annotate nugget / sill / effective range
        plt.axhline(metrics["sill"], linestyle="--", alpha=0.5)
        if not np.isnan(metrics["eff_range_km"]):
            plt.axvline(metrics["eff_range_km"], linestyle="--", alpha=0.5)
        plt.tight_layout()

        if g:
            filepath = os.path.join(self.ckpt_dir, f"sensitivity_semivariogram_global.png")
        else:
            if not os.path.exists(os.path.join(self.ckpt_dir, "local_sensitivity_variograms")):
                os.mkdir(os.path.join(self.ckpt_dir, "local_sensitivity_variograms"))
            filepath = os.path.join(self.ckpt_dir, "local_sensitivity_variograms", f"sensitivity_semivariogram_{self.baseline_coords}.png")        
        plt.savefig(filepath)
        plt.clf()


    def summarize_variogram(self, threshold=0.002):
        """
        Print a summary of variogram results in a 'stats test' style.
        
        Args:
            df (pd.DataFrame): Must contain ['distance_km', 'gamma'] columns.
            threshold (float): A heuristic cutoff for 'high fragility'.
        """
        df = self.df
        # Extract key values
        max_gamma = df['gamma'].max()
        max_dist = df.loc[df['gamma'].idxmax(), 'distance_km']
    
        # Find "elbow" distance: first distance where gamma > threshold
        elbow = df.loc[df['gamma'] > threshold, 'distance_km']
        elbow_dist = elbow.iloc[0] if not elbow.empty else None
    
        print("=" * 60)
        print(" SIMBA Local Variogram Results")
        print("=" * 60)
        print(f" Max semivariance (γ): {max_gamma:.4f} at {max_dist} km")
        if elbow_dist:
            print(f" Fragility threshold crossed at ≈ {elbow_dist} km "
                  f"(γ > {threshold})")
        else:
            print(" No fragility threshold crossed within tested distances.")
        print("-" * 60)
    
        # Baseline interpretation
        if max_gamma < threshold:
            interp = "Model is stable across all tested perturbations."
        elif elbow_dist and elbow_dist <= 10:
            interp = ("⚠️ Model highly sensitive: predictions change notably "
                      "even under small (≤10 km) spatial imprecision.")
        else:
            interp = ("Model robust locally (<10 km), but fragile under larger "
                      "perturbations.")
        print(f" Interpretation: {interp}")
        print("=" * 60)


    # -------------------------------------------------------------
    # Null baseline: bootstrapped γ(h) under coordinate jitter only
    # -------------------------------------------------------------
    @torch.no_grad()
    def simba_variogram_with_null(self,
        # model,
        # image_tensor,
        # coord_lat,
        # coord_lon,
        # device="cuda",
        distances_km = None,
        samples_per_h=16, # CHANGE THIS BACK TO 16!!
        # --- Null model params ---
        null_sigma_km=1.0,         # GPS noise scale (Rayleigh σ, in km)
        n_boot=200,                # number of bootstrap replicates
        null_dist_kind="rayleigh", # currently supports 'rayleigh'
        eff_level=0.95,
        extra_inputs: dict | None = None,
    ):
        """
        Returns:
          result: dict with observed curve, null mean/quantiles, p-values,
                  and summary stats (nugget, slope0, sill, eff_range).
        """

        
        
        # 1) Observed gamma(h)
        hs, gamma_obs, base_pred = self.df["distance_km"].to_list(), gamma_matrix_from_table(self.df)[0], self.baseline_pred

        hs = np.array(distances_km)
        
        image_tensor = self.baseline_image.to(self.device).unsqueeze(0).detach().requires_grad_(True) 
        coord_lat = self.baseline_coords[0] # DOUBLE CHECK THIS IS THE CORRECT ORDER!!
        coord_lon = self.baseline_coords[1] # DOUBLE CHECK THIS IS THE CORRECT ORDER!!
        
        # = _observed_gamma_curve(
        #     model, image_tensor, coord_lat, coord_lon,
        #     distances_km=distances_km, samples_per_h=samples_per_h,
        #     device=device, extra_inputs=extra_inputs
        # )
    
        # 2) Null baseline: for each h, draw jitter distances from the null
        #    distribution *truncated at h*, average semivariance across samples.
        rng = np.random.default_rng()
        null_mat = np.zeros((n_boot, len(hs)), dtype=float)
    
        for j, h in enumerate(hs):
            for b in range(n_boot):
                if null_dist_kind == "rayleigh":
                    dists = rayleigh_truncated(max_h=h, sigma=null_sigma_km,
                                               size=samples_per_h, rng=rng)
                else:
                    raise ValueError("Unsupported null_dist_kind")
    
                preds = []
                for d in dists:
                    lat, lon = move_from(coord_lat, coord_lon, float(d), sample_bearing())
                    c = torch.tensor([[lat, lon]], dtype=torch.float32, device=self.device)
                    if extra_inputs:
                        yhat = self.model(image_tensor.to(self.device), c,
                                     **{k: v.to(self.device) for k,v in extra_inputs.items()})
                    else:
                        yhat = self.model(image_tensor.to(self.device), c)
                    preds.append(yhat.squeeze().item())
                preds = np.array(preds)
                null_mat[b, j] = 0.5 * np.mean((preds - base_pred.detach().cpu().numpy())**2)
    
        # 3) Null summaries & p-values (one-sided: null >= observed)
        null_mean = null_mat.mean(axis=0)
        null_q95  = np.quantile(null_mat, 0.95, axis=0)

        # print(null_mat.shape)
        # print(gamma_obs)
        # print(gamma_obs.shape)

        p_per_offset, p_agg = pvalues_null_vs_obs(null_mat, gamma_obs, reduce_over_offsets="mean")


        gamma_obs_mean = self.df.groupby(self.df["distance_km"]).mean().reset_index()["gamma"].values
        
        # Package
        return {
            "distances_km": hs,                 # (D,)
            "base_pred": base_pred,
            "gamma_obs": gamma_obs,             # (K, D)
            "gamma_obs_mean": gamma_obs_mean,   # (D,)
            "null_mat": null_mat,               # (B, D)
            "null_mean": null_mat.mean(axis=0), # (D,)
            "null_q95":  np.quantile(null_mat, 0.95, axis=0),  # (D,)
            "p_per_offset": p_per_offset,       # (K, D)
            "p_per_distance": p_agg,            # (D,)
            "nugget": self.metrics["nugget"],
            "slope0": self.metrics["slope0"],
            "sill": self.metrics["sill"],
            "eff_range_km": self.metrics["eff_range_km"],
            "null_sigma_km": float(null_sigma_km),
            "n_boot": n_boot,
            "samples_per_h": samples_per_h,
            # "reduce_over_offsets": reduce_over_offsets,
            "eff_level": eff_level,
        }

        
    # -------------------------------------------
    # Optional: terminal-style summary / printout
    # -------------------------------------------
    def print_null_variogram_summary(self, results, alpha=0.05):
        """
        Pretty-print a SIMBA null variogram summary.
    
        Args:
            results (dict): Output of simba_variogram_with_null(...)
            alpha (float): Significance threshold for interpretation
        """
        hs = results["distances_km"]
        gamma_obs_mean = results["gamma_obs_mean"]
        null_mean = results["null_mean"]
        null_q95 = results["null_q95"]
        pvals = results["p_per_distance"]
    
        output_lines = []
        
        output_lines.append("\n=== SIMBA Variogram + Null Baseline Summary ===\n")
        output_lines.append(f"Base prediction: {results['base_pred']:.4f}")
        output_lines.append(f"Nugget: {results['nugget']:.6f}")
        output_lines.append(f"Slope at origin: {results['slope0']:.6f}")
        output_lines.append(f"Sill: {results['sill']:.6f}")
        output_lines.append(f"Effective range (km): {results['eff_range_km']:.2f}")
        output_lines.append(f"Bootstrap reps: {results['n_boot']}, Samples per h: {results['samples_per_h']}")
        output_lines.append(f"Null jitter σ (km): {results['null_sigma_km']:.2f}\n")
        
        header = f"{'Dist (km)'} | {'Obs γ'} | {'Null mean'} | {'Null q95'} | {'p-val'} | {'Flag'}"

        output_lines.append(header)
        output_lines.append("-" * len(header))
        
        for j, h in enumerate(hs):
            obs = gamma_obs_mean[j]
            nmean = null_mean[j]
            nq95 = null_q95[j]
            p = pvals[j]
            flag = "SIG" if p < alpha else ""
            output_lines.append(f"{h} | {obs} | {nmean} | {nq95} | {p} | {flag}")
        
        output_lines.append("\nInterpretation:")
        output_lines.append(f"- Rows marked 'SIG' mean observed semivariance > null baseline at α={alpha}.")
        output_lines.append("- This suggests the model’s predictions are more sensitive to coordinate perturbations")
        output_lines.append("  than would be expected from random GPS jitter at those distances.")
        output_lines.append("- Distances without SIG are within what could be explained by chance jitter.\n")
        
        # Join into a single string
        final_output = "\n".join(output_lines)
        
        # Print to console
        print(final_output)


        if not os.path.exists(os.path.join(self.ckpt_dir, "local_variogram_summaries")):
            os.mkdir(os.path.join(self.ckpt_dir, "local_variogram_summaries"))
        filepath = os.path.join(self.ckpt_dir, "local_variogram_summaries", f"vs_{self.baseline_coords}.txt")        

        # Save to text file
        with open(filepath, "w") as f:
            f.write(final_output)

    
        
        
        
            
            
            
            
            
