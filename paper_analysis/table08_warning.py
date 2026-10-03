"""Table 8: event-based warning verification for moderate-drought alerts (OR rule, level D1), single cell.
Input: results/dss_validation.csv."""
import pandas as pd
from _paths import RES, OUT
v = pd.read_csv(RES / "dss_validation.csv"); v = v[(v.logic == "or") & (v.level == "D1")]
T = v[["model", "horizon", "POD", "FAR", "CSI", "freq_bias", "base_rate"]].round(2).sort_values(["model", "horizon"])
T.to_csv(OUT / "table08_warning_single_cell.csv", index=False); print(T.to_string(index=False))
