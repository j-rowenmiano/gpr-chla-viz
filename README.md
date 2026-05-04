# Laguna de Bay MCI* Reconstruction Viewer

Streamlit web application for spatio-temporal visualization of GPR-reconstructed satellite-derived chlorophyll-a proxy (MCI*) in Laguna de Bay, Philippines, 2021–2025.

## Repository Structure

```
app.py
requirements.txt
data/
    gpr_reconstruction.csv
    laguna_lake.shp
    laguna_lake.dbf
    laguna_lake.shx
    north_arrow.jpg
```

## Data Requirements

| File | Description |
|------|-------------|
| `gpr_reconstruction.csv` | GPR output with columns: `cell_id`, `lon`, `lat`, `year`, `month`, `time_index`, `MCI_pred_mean`, `MCI_pred_cv_pct` |
| `laguna_lake.shp/dbf/shx` | Lake boundary shapefile (PolygonZ, 19 interior rings). No `.prj` needed — EPSG:4326 is assigned explicitly. |
| `north_arrow.jpg` | North arrow image displayed beside map |

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push this repository (with `data/` folder populated) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **New app**.
3. Connect your GitHub repo, set the main file to `app.py`, and deploy.

No secrets or environment variables required.

## Features

- **Two tabs**: MCI* Reconstruction and Predictive Uncertainty (CV%)
- **Time-lapse player**: Play/Pause, frame scrubber, variable FPS (0.5–6)
- **Consistent color scale**: Global 5th–95th percentile for MCI*, 0–99th percentile for CV%
- **Colormaps**: `YlGnBu_r` for MCI*, `Greys` for uncertainty
- **Map elements**: Lake boundary, island interiors (tan fill), scale bar (10 km), north arrow
- **Summary statistics**: Mean, median, Q25, Q75, Std Dev per month
- **Sidebar**: Month/year navigator, PNG download, stats table
- **Caching**: `@st.cache_data` keyed on `time_index` × `map_type` — frames render once and are reused

## Technical Notes

- Grid cells rendered as `matplotlib.patches.Rectangle` (0.009° × 0.009°) via `PatchCollection`, clipped to lake compound path
- PolygonZ coordinates handled with `[:, :2]` slicing throughout
- Shapefile read with `SHAPE_RESTORE_SHX=YES` and `allow_override=True` (no `.prj` file)
- Scale bar length computed from degrees longitude at latitude 14.35°N
