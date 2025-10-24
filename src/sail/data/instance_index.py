import json
import numpy as np
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088

class InstanceIndex:
    def __init__(self, json_path):
        # json looks like: {"imgA": [lat, lon], "imgB": [lat, lon], ...}
        with open(json_path, "r") as f:
            data = json.load(f)
        self.names = np.array(list(data.keys()))
        coords_deg = np.array(list(data.values()), dtype=float)[:, ::-1] # shape (N, 2) [lat, lon]
        
        # store both deg and rad if you like
        self.coords_deg = coords_deg
        self.coords_rad = np.radians(coords_deg[:, [0,1]])       # [lat, lon] -> radians
        # BallTree expects [lat, lon] in radians for haversine
        self.tree = BallTree(self.coords_rad, metric="haversine")

    def query_within_radii(self, q_lat, q_lon, radii_km=(5.0, 10.0, 15.0), return_dist=True):
        EARTH_RADIUS_KM = 6371.0088
        q = np.radians([[q_lat, q_lon]])
        out = {}
    
        for r in radii_km:
            r_rad = r / EARTH_RADIUS_KM
            # sort_results=True returns sorted arrays, still nested
            idxs_arr, dists_arr = self.tree.query_radius(
                q, r_rad, return_distance=True, sort_results=True
            )
            # unwrap the single query's results
            idxs = idxs_arr[0]          # -> np.ndarray of ints, shape (M,)
            dists_rad = dists_arr[0]    # -> np.ndarray of floats, shape (M,)
    
            if idxs.size == 0:
                out[r] = []
                continue
    
            dists_km = dists_rad * EARTH_RADIUS_KM
            out[r] = [
                (
                    self.names[i],
                    float(self.coords_deg[i, 0]),  # lat
                    float(self.coords_deg[i, 1]),  # lon
                    float(d_km),
                )
                for i, d_km in zip(idxs, dists_km)
            ]
        return out


    def query_annulus_bins(self, q_lat, q_lon, bins_km=(0,1,5,10,25)):
        """
        Optional: returns non-overlapping bins (0–1], (1–5], (5–10], ...
        """
        q = np.radians([[q_lat, q_lon]])
        # get max radius once
        max_r = max(bins_km)
        idx_all, dist_all = self.tree.query_radius(q, max_r / EARTH_RADIUS_KM, return_distance=True)
        idx_all = idx_all[0]
        dist_km = dist_all[0] * EARTH_RADIUS_KM
        # bucket
        result = {}
        for lo, hi in zip(bins_km[:-1], bins_km[1:]):
            mask = (dist_km > lo) & (dist_km <= hi)
            sel = idx_all[mask]
            dsel = dist_km[mask]
            order = np.argsort(dsel)
            sel = sel[order]; dsel = dsel[order]
            result[(lo, hi)] = [(self.names[i], float(self.coords_deg[i,0]), float(self.coords_deg[i,1]), float(d))
                                for i, d in zip(sel, dsel)]
        return result

    def nearest(self, q_lat: float, q_lon: float, top_k: int = 1):
        """
        Return the nearest image(s) to (q_lat, q_lon).
        Output: list of (name, lat, lon, dist_km) sorted by distance.
        """
        import numpy as np
        q = np.radians([[q_lat, q_lon]])  # query point in radians
        dists_rad, idxs = self.tree.query(q, k=top_k)
        dists_km = (dists_rad[0] * EARTH_RADIUS_KM).astype(float)
        idxs = idxs[0]

        # Pair up and sort (should already be sorted, but keep it explicit)
        order = np.argsort(dists_km)
        idxs = idxs[order]
        dists_km = dists_km[order]

        results = [
            (
                self.names[i],
                float(self.coords_deg[i, 0]),  # lat
                float(self.coords_deg[i, 1]),  # lon
                float(d_km),
            )
            for i, d_km in zip(idxs, dists_km)
        ]
        return results



