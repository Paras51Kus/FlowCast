from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FlowCast | Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
    ======================================================== */

    .stApp {
        background: #111214;
        color: #E5E7EB;
        font-family: Inter, Arial, sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background: #111214;
    }

    [data-testid="stHeader"] {
        background: #111214;
    }

    [data-testid="stToolbar"] {
        right: 20px;
    }

    .block-container {
        padding-top: 28px;
        padding-bottom: 30px;
        max-width: 1600px;
    }


    /* ========================================================
       SIDEBAR
    ======================================================== */

    [data-testid="stSidebar"] {
        background: #0D0E10;
        border-right: 1px solid #25282D;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 20px;
    }

    .sidebar-brand {
        padding: 8px 12px 20px 12px;
        border-bottom: 1px solid #25282D;
        margin-bottom: 22px;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .brand-icon {
        font-size: 28px;
    }

    .brand-title {
        font-size: 22px;
        font-weight: 700;
        color: #F3F4F6;
        margin: 0;
    }

    .brand-subtitle {
        font-size: 14px;
        color: #9CA3AF;
        margin-top: 3px;
    }

    .section-title {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #8B93A1;
        margin: 18px 8px 8px 8px;
    }


    /* ========================================================
       SIDEBAR BUTTONS
    ======================================================== */

    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        background: transparent;
        color: #C7CBD1;
        border: none;
        border-radius: 9px;
        padding: 11px 14px;
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 3px;
    }

    [data-testid="stSidebar"] .stButton button:hover {
        background: #171A1F;
        color: #FFFFFF;
    }

    .active-menu {
        background: #15355D;
        border-radius: 9px;
        padding: 11px 14px;
        color: #9EC5FF;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 3px;
    }


    /* ========================================================
       MAIN HEADER
    ======================================================== */

    .breadcrumb {
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #8FAFD6;
        margin-bottom: 7px;
    }

    .page-title {
        font-size: 29px;
        font-weight: 700;
        color: #F1F3F5;
        margin: 0;
        line-height: 1.2;
    }

    .page-subtitle {
        font-size: 17px;
        font-weight: 500;
        color: #A1A6AE;
        margin-top: 5px;
        margin-bottom: 16px;
    }

    .header-line {
        height: 1px;
        width: 100%;
        background: #2A2D31;
        margin-bottom: 18px;
    }


    /* ========================================================
       KPI CARDS
    ======================================================== */

    .metric-card {
        background: #1A1B1D;
        border: 1px solid #30343A;
        border-radius: 16px;
        padding: 26px 20px;
        min-height: 158px;
        transition: 0.2s ease;
    }

    .metric-card:hover {
        border-color: #3F6FA6;
        transform: translateY(-2px);
    }

    .metric-label {
        font-size: 14px;
        font-weight: 700;
        color: #9298A1;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .metric-value {
        font-size: 31px;
        font-weight: 700;
        color: #E8EAED;
        line-height: 1.1;
    }

    .metric-subtitle {
        font-size: 15px;
        color: #8F949C;
        margin-top: 7px;
    }


    /* ========================================================
       CHART CARDS
    ======================================================== */

    .chart-title {
        font-size: 18px;
        font-weight: 700;
        color: #E2E5E9;
        margin-bottom: 2px;
    }

    .chart-subtitle {
        font-size: 15px;
        color: #9298A1;
        margin-bottom: 12px;
    }

    .chart-card {
        background: #1A1B1D;
        border: 1px solid #30343A;
        border-radius: 16px;
        padding: 20px 18px 10px 18px;
        min-height: 420px;
    }


    /* ========================================================
       LEGEND
    ======================================================== */

    .legend-container {
        display: flex;
        flex-wrap: wrap;
        gap: 18px;
        margin-top: 10px;
        margin-bottom: 5px;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 7px;
        color: #B4B7BC;
        font-size: 14px;
        font-weight: 600;
    }

    .legend-dot {
        width: 14px;
        height: 14px;
        border-radius: 3px;
        display: inline-block;
    }


    /* ========================================================
       DATA SECTION
    ======================================================== */

    .data-card {
        background: #1A1B1D;
        border: 1px solid #30343A;
        border-radius: 16px;
        padding: 20px;
        margin-top: 20px;
    }

    /* ========================================================
       STREAMLIT DATAFRAME
    ======================================================== */

    [data-testid="stDataFrame"] {
        border: 1px solid #30343A;
        border-radius: 10px;
        overflow: hidden;
    }


    /* ========================================================
       RESPONSIVE
    ======================================================== */

    @media (max-width: 900px) {

        .page-title {
            font-size: 24px;
        }

        .metric-value {
            font-size: 26px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATASET SEARCH
# ============================================================

def find_dataset():

    possible_files = [
        BASE_DIR / "data" / "processed" / "flowcast_features.csv",
        BASE_DIR / "data" / "processed" / "traffic_features.csv",
        BASE_DIR / "data" / "flowcast_features.csv",
        BASE_DIR / "data" / "traffic_data.csv",
        BASE_DIR / "dataset" / "flowcast_features.csv",
        BASE_DIR / "flowcast_features.csv",
    ]

    for file_path in possible_files:
        if file_path.exists():
            return file_path

    # Search entire project
    possible_names = [
        "flowcast_features.csv",
        "traffic_features.csv",
        "traffic_data.csv",
        "processed_data.csv",
    ]

    for name in possible_names:

        found_files = list(BASE_DIR.rglob(name))

        if found_files:
            return found_files[0]

    return None


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    dataset_path = find_dataset()

    if dataset_path is None:
        return pd.DataFrame()

    try:

        dataframe = pd.read_csv(dataset_path)

        return dataframe

    except Exception as error:

        st.error(f"Error loading dataset: {error}")

        return pd.DataFrame()


df = load_data()


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(dataframe, possible_names):

    if dataframe.empty:
        return None

    column_map = {
        column.lower().strip(): column
        for column in dataframe.columns
    }

    for name in possible_names:

        name = name.lower().strip()

        if name in column_map:
            return column_map[name]

    return None


# ============================================================
# DETECT COLUMNS
# ============================================================

datetime_column = find_column(
    df,
    [
        "datetime",
        "timestamp",
        "date_time",
        "date",
        "time",
    ],
)

road_column = find_column(
    df,
    [
        "road_name",
        "road",
        "road_segment",
        "segment",
        "location",
    ],
)

volume_column = find_column(
    df,
    [
        "traffic_volume",
        "vehicle_count",
        "volume",
        "vehicles",
        "traffic_count",
        "vehicle_volume",
    ],
)

speed_column = find_column(
    df,
    [
        "avg_speed",
        "average_speed",
        "speed",
        "vehicle_speed",
    ],
)

congestion_column = find_column(
    df,
    [
        "congestion_level",
        "congestion",
        "traffic_level",
        "congestion_status",
    ],
)


# ============================================================
# DATA PREPROCESSING
# ============================================================

processed_df = df.copy()

if not processed_df.empty:

    # Convert datetime
    if datetime_column:

        processed_df[datetime_column] = pd.to_datetime(
            processed_df[datetime_column],
            errors="coerce",
        )

    # Convert numeric columns
    if volume_column:

        processed_df[volume_column] = pd.to_numeric(
            processed_df[volume_column],
            errors="coerce",
        )

    if speed_column:

        processed_df[speed_column] = pd.to_numeric(
            processed_df[speed_column],
            errors="coerce",
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <div class="brand-row">

                <div class="brand-icon">
                    🚦
                </div>

                <div>

                    <div class="brand-title">
                        FlowCast
                    </div>

                    <div class="brand-subtitle">
                        Traffic Intelligence
                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- ANALYTICS ----------------

    st.markdown(
        '<div class="section-title">ANALYTICS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="active-menu">
            ▦ &nbsp; Overview
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.button("⌁  Historical Trends", use_container_width=True)
    st.button("⠿  Congestion Map", use_container_width=True)
    st.button("╱╲  Road Comparison", use_container_width=True)
    st.button("☁  Weather vs Traffic", use_container_width=True)

    # ---------------- PREDICTIONS ----------------

    st.markdown(
        '<div class="section-title">PREDICTIONS</div>',
        unsafe_allow_html=True,
    )

    st.button("▷  Live Prediction", use_container_width=True)
    st.button("⌁  Forecast View", use_container_width=True)
    st.button("⬡  Confidence Band", use_container_width=True)

    # ---------------- SYSTEM ----------------

    st.markdown(
        '<div class="section-title">SYSTEM</div>',
        unsafe_allow_html=True,
    )

    st.button("⚙  Model Performance", use_container_width=True)
    st.button("⇧  Data Upload", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not processed_df.empty:

        st.caption(
            f"Dataset loaded: {len(processed_df):,} records"
        )

    else:

        st.caption("No dataset loaded")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="breadcrumb">
        🚦 &nbsp; FLOWCAST / OVERVIEW
    </div>

    <div class="page-title">
        Traffic Overview
    </div>

    <div class="page-subtitle">
        Real-time snapshot of network-wide flow metrics
    </div>

    <div class="header-line"></div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EMPTY DATA STATE
# ============================================================

if processed_df.empty:

    st.warning(
        """
        No FlowCast dataset found.

        Make sure your dataset is located at:

        data/processed/flowcast_features.csv
        """
    )

    st.stop()


# ============================================================
# CALCULATE METRICS
# ============================================================

total_records = len(processed_df)


# Road segments
if road_column:

    road_segments = processed_df[
        road_column
    ].nunique()

else:

    road_segments = 0


# Average traffic volume
if volume_column:

    average_volume = processed_df[
        volume_column
    ].mean()

    if pd.isna(average_volume):
        average_volume = 0

else:

    average_volume = 0


# Average speed
if speed_column:

    average_speed = processed_df[
        speed_column
    ].mean()

    if pd.isna(average_speed):
        average_speed = 0

else:

    average_speed = 0


# ============================================================
# KPI CARDS
# ============================================================

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)


with metric_col1:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Total Records
            </div>

            <div class="metric-value">
                {total_records:,.0f}
            </div>

            <div class="metric-subtitle">
                across all segments
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with metric_col2:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Road Segments
            </div>

            <div class="metric-value">
                {road_segments:,.0f}
            </div>

            <div class="metric-subtitle">
                active sensors
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with metric_col3:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Avg Volume
            </div>

            <div class="metric-value">
                {average_volume:,.0f}
            </div>

            <div class="metric-subtitle">
                vehicles / window
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with metric_col4:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-label">
                Avg Speed
            </div>

            <div class="metric-value">
                {average_speed:.1f}
            </div>

            <div class="metric-subtitle">
                km/h network-wide
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SPACE
# ============================================================

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


# ============================================================
# CHART LAYOUT
# ============================================================

chart_col1, chart_col2 = st.columns([1.65, 1])


# ============================================================
# HOURLY VOLUME PATTERN
# ============================================================

with chart_col1:

    st.markdown(
        """
        <div class="chart-card">

            <div class="chart-title">
                Hourly volume pattern
            </div>

            <div class="chart-subtitle">
                Network-wide average across all road segments
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Create hourly data

    if datetime_column and volume_column:

        chart_df = processed_df.dropna(
            subset=[
                datetime_column,
                volume_column,
            ]
        ).copy()

        if not chart_df.empty:

            chart_df["hour"] = chart_df[
                datetime_column
            ].dt.hour

            hourly_data = (
                chart_df
                .groupby("hour")[volume_column]
                .mean()
                .reset_index()
            )

            # Make sure all 24 hours exist
            full_hours = pd.DataFrame(
                {"hour": range(24)}
            )

            hourly_data = full_hours.merge(
                hourly_data,
                on="hour",
                how="left",
            )

            hourly_data[volume_column] = (
                hourly_data[volume_column]
                .interpolate()
                .bfill()
                .ffill()
                .fillna(0)
            )

        else:

            hourly_data = pd.DataFrame(
                {
                    "hour": range(24),
                    volume_column: [0] * 24,
                }
            )

    elif volume_column:

        # If datetime doesn't exist,
        # create a trend from row order

        sample_size = min(
            24,
            len(processed_df),
        )

        values = (
            processed_df[volume_column]
            .dropna()
            .head(sample_size)
            .tolist()
        )

        if len(values) < 24:

            values = values + [0] * (
                24 - len(values)
            )

        hourly_data = pd.DataFrame(
            {
                "hour": range(24),
                volume_column: values[:24],
            }
        )

    else:

        hourly_data = pd.DataFrame(
            {
                "hour": range(24),
                "volume": [0] * 24,
            }
        )

        volume_column = "volume"


    fig_hourly = go.Figure()

    fig_hourly.add_trace(

        go.Scatter(

            x=hourly_data["hour"],

            y=hourly_data[volume_column],

            mode="lines",

            line=dict(
                color="#3B82D0",
                width=3,
                shape="spline",
                smoothing=1.1,
            ),

            fill="tozeroy",

            fillcolor="rgba(59,130,208,0.08)",

            hovertemplate=(
                "Hour: %{x}:00"
                "<br>Volume: %{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )


    fig_hourly.update_layout(

        height=300,

        margin=dict(
            l=45,
            r=15,
            t=10,
            b=35,
        ),

        paper_bgcolor="#1A1B1D",

        plot_bgcolor="#1A1B1D",

        showlegend=False,

        hovermode="x unified",

        xaxis=dict(

            title=None,

            tickmode="array",

            tickvals=[
                0,
                3,
                6,
                9,
                12,
                15,
                18,
                21,
            ],

            ticktext=[
                "0:00",
                "3:00",
                "6:00",
                "9:00",
                "12:00",
                "15:00",
                "18:00",
                "21:00",
            ],

            showgrid=True,

            gridcolor="#30343A",

            zeroline=False,

            color="#9BA1A9",
        ),

        yaxis=dict(

            title=None,

            showgrid=True,

            gridcolor="#30343A",

            zeroline=False,

            color="#9BA1A9",

            tickformat=",",
        ),
    )


    st.plotly_chart(
        fig_hourly,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# CONGESTION SPLIT
# ============================================================

with chart_col2:

    st.markdown(
        """
        <div class="chart-card">

            <div class="chart-title">
                Congestion split
            </div>

            <div class="chart-subtitle">
                Share of time in each congestion state
            </div>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # PREPARE CONGESTION DATA
    # --------------------------------------------------------

    if congestion_column:

        congestion_data = (
            processed_df[congestion_column]
            .astype(str)
            .str.strip()
            .value_counts()
        )

        labels = congestion_data.index.tolist()

        values = congestion_data.values.tolist()

    else:

        labels = [
            "Free-flow",
            "Moderate",
            "Severe",
        ]

        values = [
            60,
            28,
            12,
        ]


    # Normalize names

    display_labels = []

    for label in labels:

        label_lower = label.lower()

        if (
            "free" in label_lower
            or "low" in label_lower
            or "light" in label_lower
        ):

            display_labels.append(
                "Free-flow"
            )

        elif (
            "moderate" in label_lower
            or "medium" in label_lower
        ):

            display_labels.append(
                "Moderate"
            )

        elif (
            "severe" in label_lower
            or "high" in label_lower
            or "heavy" in label_lower
        ):

            display_labels.append(
                "Severe"
            )

        else:

            display_labels.append(
                str(label)
            )


    # Calculate percentages

    total_congestion = sum(values)

    if total_congestion == 0:
        total_congestion = 1


    percentages = [
        (value / total_congestion) * 100
        for value in values
    ]


    # Legend

    color_map = {
        "Free-flow": "#36B89C",
        "Moderate": "#FFB000",
        "Severe": "#F05454",
    }


    legend_html = '<div class="legend-container">'


    for label, percentage in zip(
        display_labels,
        percentages,
    ):

        color = color_map.get(
            label,
            "#3B82D0",
        )

        legend_html += f"""
        <div class="legend-item">

            <span
                class="legend-dot"
                style="background:{color};"
            ></span>

            {label} {percentage:.0f}%

        </div>
        """


    legend_html += "</div>"


    st.markdown(
        legend_html,
        unsafe_allow_html=True,
    )


    colors = []

    for label in display_labels:

        colors.append(
            color_map.get(
                label,
                "#3B82D0",
            )
        )


    fig_congestion = go.Figure(

        data=[
            go.Pie(

                labels=display_labels,

                values=values,

                hole=0.64,

                marker=dict(
                    colors=colors,
                    line=dict(
                        color="#1A1B1D",
                        width=0,
                    ),
                ),

                textinfo="none",

                hovertemplate=(
                    "%{label}"
                    "<br>%{percent}"
                    "<extra></extra>"
                ),
            )
        ]
    )


    fig_congestion.update_layout(

        height=275,

        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0,
        ),

        paper_bgcolor="#1A1B1D",

        plot_bgcolor="#1A1B1D",

        showlegend=False,
    )


    st.plotly_chart(
        fig_congestion,
        use_container_width=True,
        config={
            "displayModeBar": False,
        },
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# LOWER SECTION
# ============================================================

st.markdown(
    "<div style='height:10px'></div>",
    unsafe_allow_html=True,
)


# ============================================================
# ROAD PERFORMANCE TABLE
# ============================================================

st.markdown(
    """
    <div class="data-card">

        <div class="chart-title">
            Road Network Summary
        </div>

        <div class="chart-subtitle">
            Performance overview across monitored road segments
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BUILD SUMMARY TABLE
# ============================================================

if road_column:

    aggregation = {}

    if volume_column:
        aggregation[volume_column] = "mean"

    if speed_column:
        aggregation[speed_column] = "mean"


    if aggregation:

        summary_df = (

            processed_df

            .groupby(road_column)

            .agg(aggregation)

            .reset_index()

        )

        rename_columns = {
            road_column: "Road Segment",
        }

        if volume_column:
            rename_columns[
                volume_column
            ] = "Average Volume"

        if speed_column:
            rename_columns[
                speed_column
            ] = "Average Speed"


        summary_df = summary_df.rename(
            columns=rename_columns
        )


        if "Average Volume" in summary_df.columns:

            summary_df[
                "Average Volume"
            ] = summary_df[
                "Average Volume"
            ].round(0)


        if "Average Speed" in summary_df.columns:

            summary_df[
                "Average Speed"
            ] = summary_df[
                "Average Speed"
            ].round(1)


        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            height=280,
        )

    else:

        st.dataframe(
            processed_df.head(20),
            use_container_width=True,
            hide_index=True,
            height=280,
        )

else:

    st.dataframe(
        processed_df.head(20),
        use_container_width=True,
        hide_index=True,
        height=280,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        border-top: 1px solid #2A2D31;
        margin-top: 35px;
        padding-top: 18px;
        padding-bottom: 10px;
        text-align: center;
        color: #737982;
        font-size: 13px;
    ">

        FlowCast Traffic Intelligence System

        &nbsp; • &nbsp;

        Data-driven traffic analytics

    </div>
    """,
    unsafe_allow_html=True,
)