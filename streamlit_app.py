"""
Laguna de Bay MCI* Reconstruction & Uncertainty Viewer
Streamlit web application for spatio-temporal chlorophyll-a proxy time-lapse.
"""

import os
import io
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize, BoundaryNorm
import matplotlib.cm as cm
import geopandas as gpd
import streamlit as st
from shapely.geometry import box

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Laguna de Bay · MCI* Viewer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
      font-family: 'IBM Plex Sans', sans-serif;
  }
  h1, h2, h3 {
      font-family: 'IBM Plex Mono', monospace;
      letter-spacing: -0.02em;
  }
  .stTabs [data-baseweb="tab-list"] {
      gap: 8px;
      border-bottom: 2px solid #1a3a4a;
  }
  .stTabs [data-baseweb="tab"] {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.85rem;
      font-weight: 600;
      color: #5a8fa8;
      border-radius: 4px 4px 0 0;
      padding: 8px 20px;
      background: transparent;
  }
  .stTabs [aria-selected="true"] {
      color: #e8f4f8 !important;
      background: #1a3a4a !important;
  }
  .stat-card {
      background: #0e2330;
      border: 1px solid #1e4060;
      border-radius: 6px;
      padding: 10px 14px;
      text-align: center;
  }
  .stat-label {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.68rem;
      color: #6aaccc;
      text-transform: uppercase;
      letter-spacing: 0.08em;
  }
  .stat-value {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 1.1rem;
      font-weight: 600;
      color: #c8e8f4;
  }
  .frame-title {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 1.3rem;
      font-weight: 600;
      color: #c8e8f4;
      letter-spacing: 0.04em;
  }
  .sidebar-header {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.75rem;
      color: #6aaccc;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      padding-bottom: 4px;
      border-bottom: 1px solid #1e4060;
      margin-bottom: 10px;
  }
  /* Dark background for the whole app */
  .stApp {
      background-color: #060f17;
  }
  section[data-testid="stSidebar"] {
      background-color: #091520;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "gpr_reconstruction.csv")
SHP_PATH = os.path.join(DATA_DIR, "laguna_lake.shp")
NORTH_ARROW_PATH = os.path.join(DATA_DIR, "north_arrow.jpg")
CELL_SIZE_DEG = 0.009          # 1 km grid
ISLAND_COLOR = "#c8b89a"
LAKE_EDGE_COLOR = "#1a2a3a"
SCALE_BAR_LAT = 14.35          # reference latitude for scale bar
SCALE_BAR_KM = 10
N_MONTHS = 60
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "playing": False,
    "frame_idx": 0,
    "fps": 2,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values(["time_index", "cell_id"]).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_shapefile():
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    gdf = gpd.read_file(SHP_PATH, engine="pyogrio")
    gdf = gdf.set_crs(epsg=4326, allow_override=True)
    return gdf

@st.cache_data(show_spinner=False)
def compute_global_limits(df):
    mci_vmin = np.nanpercentile(df["MCI_pred_mean"], 5)
    mci_vmax = np.nanpercentile(df["MCI_pred_mean"], 95)
    cv_vmin = 0.0
    cv_vmax = np.nanpercentile(df["MCI_pred_cv_pct"], 99)
    return mci_vmin, mci_vmax, cv_vmin, cv_vmax

@st.cache_data(show_spinner=False)
def build_clip_path(gdf):
    """
    Build a compound matplotlib Path for the lake exterior minus island interiors.
    Handles PolygonZ by slicing [:, :2].
    """
    geom = gdf.geometry.iloc[0]
    # Collect all rings (exterior + interiors)
    paths = []
    ext_coords = np.array(geom.exterior.coords)[:, :2]
    paths.append(Path(ext_coords))
    for interior in geom.interiors:
        int_coords = np.array(interior.coords)[:, :2]
        paths.append(Path(int_coords))
    compound = Path.make_compound_path(*paths)
    return compound

@st.cache_data(show_spinner=False)
def get_month_data(df, time_index):
    return df[df["time_index"] == time_index].copy()

@st.cache_data(show_spinner=False)
def render_frame(time_index: int, map_type: str,
                 mci_vmin, mci_vmax, cv_vmin, cv_vmax):
    """
    Render a single frame as a PNG byte buffer.
    Cached by time_index + map_type.
    """
    df = load_data()
    gdf = load_shapefile()
    clip_path = build_clip_path(gdf)
    month_df = get_month_data(df, time_index)

    # Determine field and colormap
    if map_type == "mci":
        field = "MCI_pred_mean"
        cmap_name = "YlGnBu_r"
        vmin, vmax = mci_vmin, mci_vmax
        cbar_label = "MCI* (W m⁻² sr⁻¹ µm⁻¹)"
        title_prefix = "MCI* Reconstruction"
    else:
        field = "MCI_pred_cv_pct"
        cmap_name = "Greys"
        vmin, vmax = cv_vmin, cv_vmax
        cbar_label = "Predictive Uncertainty (CV%)"
        title_prefix = "GPR Predictive Uncertainty"

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Month/year label
    year = int(month_df["year"].iloc[0])
    month = int(month_df["month"].iloc[0])
    month_label = f"{MONTH_NAMES[month - 1]} {year}"

    # ── Figure layout ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 8), facecolor="#060f17")
    # Main axes for map
    ax = fig.add_axes([0.04, 0.10, 0.72, 0.82])
    # Colorbar axes
    cax = fig.add_axes([0.79, 0.15, 0.025, 0.65])
    # North arrow axes (top-right)
    nax = fig.add_axes([0.83, 0.78, 0.12, 0.14])

    ax.set_facecolor("#060f17")
    fig.patch.set_facecolor("#060f17")

    # ── Draw lake fill and boundary ──────────────────────────────────────
    geom = gdf.geometry.iloc[0]
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath
    import matplotlib.patches as mpa

    # Draw lake body background (water color)
    ext_coords = np.array(geom.exterior.coords)[:, :2]
    lake_bg_path = MplPath(ext_coords)
    lake_bg_patch = PathPatch(lake_bg_path, facecolor="#0a1e2e",
                              edgecolor="none", zorder=0)
    ax.add_patch(lake_bg_patch)

    # Draw island interiors filled with muted tan
    for interior in geom.interiors:
        int_coords = np.array(interior.coords)[:, :2]
        int_path = MplPath(int_coords)
        island_patch = PathPatch(int_path, facecolor=ISLAND_COLOR,
                                 edgecolor="#8a7a64", linewidth=0.5, zorder=3)
        ax.add_patch(island_patch)

    # ── Build PatchCollection of grid cells ─────────────────────────────
    half = CELL_SIZE_DEG / 2.0
    patches = []
    values = []
    for _, row in month_df.iterrows():
        rect = mpatches.Rectangle(
            (row["lon"] - half, row["lat"] - half),
            CELL_SIZE_DEG, CELL_SIZE_DEG
        )
        patches.append(rect)
        values.append(row[field])

    pc = PatchCollection(patches, cmap=cmap, norm=norm,
                         linewidth=0, zorder=2)
    pc.set_array(np.array(values))
    # Clip to lake compound path
    pc.set_clip_path(
        PathPatch(clip_path, transform=ax.transData),
    )
    ax.add_collection(pc)

    # ── Lake boundary outline on top ─────────────────────────────────────
    gdf.boundary.plot(ax=ax, color=LAKE_EDGE_COLOR, linewidth=1.2, zorder=4)

    # ── Map extent ───────────────────────────────────────────────────────
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    pad = 0.02
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
    ax.set_aspect("equal")

    # ── Axes styling ─────────────────────────────────────────────────────
    ax.tick_params(colors="#6aaccc", labelsize=7)
    ax.set_xlabel("Longitude (°E)", color="#6aaccc", fontsize=8,
                  fontfamily="monospace")
    ax.set_ylabel("Latitude (°N)", color="#6aaccc", fontsize=8,
                  fontfamily="monospace")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e4060")
    ax.grid(color="#1e4060", linewidth=0.4, linestyle="--", alpha=0.5)

    # ── Title ────────────────────────────────────────────────────────────
    ax.set_title(f"{title_prefix} · {month_label}",
                 color="#c8e8f4", fontsize=11,
                 fontfamily="monospace", fontweight="bold", pad=8)

    # ── Colorbar ─────────────────────────────────────────────────────────
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label(cbar_label, color="#c8e8f4", fontsize=7,
                 fontfamily="monospace")
    cb.ax.tick_params(colors="#c8e8f4", labelsize=6.5)
    cb.outline.set_edgecolor("#1e4060")

    # ── Scale bar ────────────────────────────────────────────────────────
    # 10 km in degrees longitude at reference latitude
    import math
    km_per_deg_lon = 111.32 * math.cos(math.radians(SCALE_BAR_LAT))
    scale_deg = SCALE_BAR_KM / km_per_deg_lon

    sb_x0 = bounds[0] + 0.03
    sb_y0 = bounds[1] + 0.015
    sb_y1 = sb_y0 + 0.008
    # Bar
    ax.plot([sb_x0, sb_x0 + scale_deg], [sb_y0, sb_y0],
            color="#c8e8f4", linewidth=2.5, zorder=5,
            solid_capstyle="butt")
    # End ticks
    for xp in [sb_x0, sb_x0 + scale_deg]:
        ax.plot([xp, xp], [sb_y0 - 0.004, sb_y0 + 0.004],
                color="#c8e8f4", linewidth=1.5, zorder=5)
    # Label
    ax.text(sb_x0 + scale_deg / 2, sb_y0 + 0.009,
            f"{SCALE_BAR_KM} km",
            ha="center", va="bottom", fontsize=7,
            color="#c8e8f4", fontfamily="monospace", zorder=5,
            path_effects=[pe.withStroke(linewidth=2, foreground="#060f17")])

    # ── North arrow ──────────────────────────────────────────────────────
    nax.set_facecolor("#060f17")
    nax.axis("off")
    if os.path.exists(NORTH_ARROW_PATH):
        from PIL import Image
        narr = Image.open(NORTH_ARROW_PATH)
        nax.imshow(narr, aspect="auto")
    else:
        # Fallback: draw a simple N arrow
        nax.annotate("N", xy=(0.5, 0.85), xytext=(0.5, 0.15),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color="#c8e8f4", lw=2),
                     ha="center", va="center", fontsize=12,
                     color="#c8e8f4", fontfamily="monospace",
                     fontweight="bold")

    # ── Save to buffer ────────────────────────────────────────────────────
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def compute_stats(month_df, field):
    vals = month_df[field].dropna()
    return {
        "Mean": vals.mean(),
        "Median": vals.median(),
        "Q25": vals.quantile(0.25),
        "Q75": vals.quantile(0.75),
        "Std Dev": vals.std(),
        "N cells": len(vals),
    }

def format_stat(key, val, map_type):
    unit = "W m⁻² sr⁻¹ µm⁻¹" if map_type == "mci" else "CV%"
    if key == "N cells":
        return f"{int(val)}"
    elif map_type == "mci":
        return f"{val:.6f}"
    else:
        return f"{val:.2f}"

def time_index_to_label(ti):
    year = 2021 + ti // 12
    month = (ti % 12) + 1
    return f"{MONTH_NAMES[month - 1]} {year}"

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset…"):
    df = load_data()
    gdf = load_shapefile()
    mci_vmin, mci_vmax, cv_vmin, cv_vmax = compute_global_limits(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:1.1rem;
         font-weight:600; color:#c8e8f4; margin-bottom:4px;">
    🌊 Laguna de Bay
    </div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
         color:#6aaccc; letter-spacing:0.08em; margin-bottom:18px;">
    MCI* · GPR RECONSTRUCTION
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-header">Navigate by Month</div>',
                unsafe_allow_html=True)
    # Month/year selector
    year_sel = st.selectbox("Year", list(range(2021, 2026)), index=0)
    month_sel = st.selectbox("Month", MONTH_NAMES, index=0)
    month_num_sel = MONTH_NAMES.index(month_sel) + 1
    nav_time_index = (year_sel - 2021) * 12 + (month_num_sel - 1)
    if st.button("Go to this month →", use_container_width=True):
        st.session_state.frame_idx = nav_time_index
        st.session_state.playing = False

    st.divider()
    st.markdown('<div class="sidebar-header">Export</div>',
                unsafe_allow_html=True)
    export_tab = st.radio("Map type to export",
                          ["MCI* Reconstruction", "Predictive Uncertainty"],
                          index=0, label_visibility="collapsed")
    export_map_type = "mci" if "MCI*" in export_tab else "cv"

    if st.button("⬇ Download current frame PNG", use_container_width=True):
        with st.spinner("Rendering…"):
            png_bytes = render_frame(
                st.session_state.frame_idx, export_map_type,
                mci_vmin, mci_vmax, cv_vmin, cv_vmax
            )
        st.download_button(
            label="Save PNG",
            data=png_bytes,
            file_name=f"laguna_{export_map_type}_{time_index_to_label(st.session_state.frame_idx).replace(' ', '_')}.png",
            mime="image/png",
            use_container_width=True,
        )

    st.divider()
    st.markdown('<div class="sidebar-header">Summary Statistics</div>',
                unsafe_allow_html=True)
    sidebar_map_type = "mci" if "MCI*" in export_tab else "cv"
    sb_month_df = get_month_data(df, st.session_state.frame_idx)
    sb_field = "MCI_pred_mean" if sidebar_map_type == "mci" else "MCI_pred_cv_pct"
    sb_stats = compute_stats(sb_month_df, sb_field)
    sidebar_label = time_index_to_label(st.session_state.frame_idx)
    st.caption(sidebar_label)
    stats_df = pd.DataFrame(
        [(k, format_stat(k, v, sidebar_map_type)) for k, v in sb_stats.items()],
        columns=["Statistic", "Value"]
    )
    st.dataframe(stats_df, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
         color:#3a6a8a; line-height:1.6;">
    Sentinel-3 OLCI · GPyTorch GPR<br>
    Kernel: ScaleRBF + ScaleMatérn 3/2<br>
    Grid: 895 cells · 1 km × 1 km<br>
    Period: Jan 2021 – Dec 2025
    </div>
    """, unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-family:'IBM Plex Mono',monospace; font-size:1.5rem;
     color:#c8e8f4; margin-bottom:2px;">
Spatio-Temporal MCI* Reconstruction
</h1>
<p style="font-family:'IBM Plex Sans',sans-serif; font-size:0.85rem;
   color:#6aaccc; margin-bottom:20px;">
Laguna de Bay, Philippines &nbsp;·&nbsp; 2021–2025 &nbsp;·&nbsp;
Gaussian Process Regression · Sentinel-3 OLCI
</p>
""", unsafe_allow_html=True)

tab_mci, tab_cv = st.tabs(["📊  MCI* Reconstruction", "🌫️  Predictive Uncertainty"])

def render_tab(map_type: str):
    field = "MCI_pred_mean" if map_type == "mci" else "MCI_pred_cv_pct"
    field_label = "MCI* (W m⁻² sr⁻¹ µm⁻¹)" if map_type == "mci" else "CV (%)"

    # ── Player controls ───────────────────────────────────────────────────
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 5, 2])
    with ctrl_col1:
        if st.session_state.playing:
            if st.button("⏸ Pause", use_container_width=True):
                st.session_state.playing = False
        else:
            if st.button("▶ Play", use_container_width=True):
                st.session_state.playing = True

    with ctrl_col2:
        frame_idx = st.slider(
            "Month",
            min_value=0,
            max_value=N_MONTHS - 1,
            value=st.session_state.frame_idx,
            format="",
            label_visibility="collapsed",
            key=f"slider_{map_type}",
        )
        st.session_state.frame_idx = frame_idx

    with ctrl_col3:
        fps = st.selectbox(
            "Speed",
            options=[0.5, 1, 2, 4, 6],
            index=2,
            format_func=lambda x: f"{x} fps",
            label_visibility="collapsed",
            key=f"fps_{map_type}",
        )
        st.session_state.fps = fps

    # Current month label
    current_label = time_index_to_label(st.session_state.frame_idx)
    st.markdown(
        f'<div class="frame-title">📅 {current_label}</div>',
        unsafe_allow_html=True
    )

    # ── Map ───────────────────────────────────────────────────────────────
    map_placeholder = st.empty()
    with map_placeholder:
        with st.spinner(f"Rendering {current_label}…"):
            png_bytes = render_frame(
                st.session_state.frame_idx, map_type,
                mci_vmin, mci_vmax, cv_vmin, cv_vmax
            )
        st.image(png_bytes, use_container_width=True)

    # ── Summary statistics inline ─────────────────────────────────────────
    month_df = get_month_data(df, st.session_state.frame_idx)
    stats = compute_stats(month_df, field)
    st.markdown("---")
    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\',monospace; '
        'font-size:0.72rem; color:#6aaccc; letter-spacing:0.08em; '
        'text-transform:uppercase; margin-bottom:8px;">'
        f'Lake-wide summary · {current_label} · {field_label}</div>',
        unsafe_allow_html=True
    )
    stat_cols = st.columns(6)
    stat_items = [
        ("Mean", stats["Mean"]),
        ("Median", stats["Median"]),
        ("Q25", stats["Q25"]),
        ("Q75", stats["Q75"]),
        ("Std Dev", stats["Std Dev"]),
        ("N Cells", stats["N cells"]),
    ]
    for col, (label, val) in zip(stat_cols, stat_items):
        display_val = format_stat(
            "N cells" if label == "N Cells" else label,
            val, map_type
        )
        col.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-label">{label}</div>'
            f'<div class="stat-value">{display_val}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Auto-advance if playing ───────────────────────────────────────────
    if st.session_state.playing:
        time.sleep(1.0 / st.session_state.fps)
        next_idx = (st.session_state.frame_idx + 1) % N_MONTHS
        st.session_state.frame_idx = next_idx
        st.rerun()

with tab_mci:
    render_tab("mci")

with tab_cv:
    render_tab("cv")
