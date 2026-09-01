from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


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
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA = BASE_DIR / "data" / "processed" / "flowcast_features.csv"
MODEL_DIR = BASE_DIR / "models" / "classical"
DEEP_MODEL_DIR = BASE_DIR / "models" / "deep_learning"
REPORT_DIR = BASE_DIR / "reports"


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_data():

    if not DATA.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(DATA)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(
                df["timestamp"],
                errors="coerce"
            )

        return df

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_volume_model():

    model_path = MODEL_DIR / "xgboost_volume.joblib"

    if not model_path.exists():
        return None

    try:
        return joblib.load(model_path)
    except Exception:
        return None


df = load_data()


# ============================================================
# PLOTLY THEME
# ============================================================

def apply_theme(fig):

    fig.update_layout(

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        font=dict(
            family="Arial, sans-serif",
            size=13
        ),

        margin=dict(
            t=50,
            b=40,
            l=50,
            r=30
        ),

        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(128,128,128,0.15)"
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(128,128,128,0.15)"
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),

        hovermode="x unified"
    )

    return fig


# ============================================================
# MODEL PREDICTION
# ============================================================

def model_prediction_frame(data):

    package = load_volume_model()

    if package is None:
        return None

    try:

        model = package["model"]
        features = package["features"]

        X = data.reindex(
            columns=features,
            fill_value=0
        )

        predictions = model.predict(X)

        return predictions

    except Exception as e:

        st.warning(f"Prediction error: {e}")

        return None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def page_header(title, subtitle=None):

    st.title(title)

    if subtitle:
        st.caption(subtitle)

    st.divider()


def show_kpis(items):
    """
    items format:

    [
        {
            "label": "Total Records",
            "value": "48,000",
            "delta": None
        }
    ]
    """

    columns = st.columns(len(items))

    for col, item in zip(columns, items):

        with col:
            with st.container(border=True):

                st.metric(
                    label=item["label"],
                    value=item["value"],
                    delta=item.get("delta")
                )


def chart_container(title, subtitle=None):

    container = st.container(border=True)

    with container:

        st.subheader(title)

        if subtitle:
            st.caption(subtitle)

        return container


def safe_mean(dataframe, column):

    if column in dataframe.columns:
        return dataframe[column].mean()

    return 0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🚦 FlowCast")

    st.caption("Traffic Intelligence Platform")

    st.divider()

    st.subheader("ANALYTICS")

    analytics_page = st.radio(

        "Analytics",

        [
            "Overview",
            "Historical Trends",
            "Congestion Heatmap",
            "Road Comparison",
            "Weather vs Traffic"
        ],

        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("PREDICTIONS")

    prediction_page = st.radio(

        "Predictions",

        [
            "Live Prediction",
            "Forecast Visualisation",
            "Prediction Confidence"
        ],

        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("SYSTEM")

    system_page = st.radio(

        "System",

        [
            "Model Performance",
            "Feature Importance",
            "Data Upload",
            "Reports"
        ],

        label_visibility="collapsed"
    )

    st.divider()

    # Navigation selector
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Overview"

    analytics_options = [
        "Overview",
        "Historical Trends",
        "Congestion Heatmap",
        "Road Comparison",
        "Weather vs Traffic"
    ]

    prediction_options = [
        "Live Prediction",
        "Forecast Visualisation",
        "Prediction Confidence"
    ]

    system_options = [
        "Model Performance",
        "Feature Importance",
        "Data Upload",
        "Reports"
    ]

    # Detect active page
    if analytics_page != "Overview":
        st.session_state.active_page = analytics_page

    if prediction_page != "Live Prediction":
        st.session_state.active_page = prediction_page

    if system_page != "Model Performance":
        st.session_state.active_page = system_page

    active = st.session_state.active_page

    if not df.empty:

        st.success("Data loaded")

        st.caption(
            f"{len(df):,} records available"
        )

    else:

        st.error("No data found")


# ============================================================
# NO DATA GUARD
# ============================================================

if df.empty:

    st.warning("No processed FlowCast dataset found.")

    st.code(
        "python run_pipeline.py",
        language="bash"
    )

    st.info(
        "Run the pipeline first to generate "
        "data/processed/flowcast_features.csv"
    )

    st.stop()


# ============================================================
# OVERVIEW
# ============================================================

if active == "Overview":

    page_header(

        "Traffic Overview",

        "Real-time snapshot of network-wide traffic metrics"
    )

    # --------------------------------------------------------
    # KPI DATA
    # --------------------------------------------------------

    records = len(df)

    roads = (
        df["road_id"].nunique()
        if "road_id" in df.columns
        else 0
    )

    avg_volume = safe_mean(
        df,
        "traffic_volume"
    )

    avg_speed = safe_mean(
        df,
        "avg_speed"
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    show_kpis([

        {
            "label": "Total Records",
            "value": f"{records:,}"
        },

        {
            "label": "Road Segments",
            "value": f"{roads:,}"
        },

        {
            "label": "Avg Traffic Volume",
            "value": f"{avg_volume:,.0f}",
            "delta": "vehicles / window"
        },

        {
            "label": "Avg Speed",
            "value": f"{avg_speed:.1f}",
            "delta": "km/h"
        }

    ])

    st.write("")

    # --------------------------------------------------------
    # HOURLY VOLUME + CONGESTION
    # --------------------------------------------------------

    col1, col2 = st.columns([3, 2])

    # --------------------------------------------------------
    # HOURLY VOLUME
    # --------------------------------------------------------

    with col1:

        with st.container(border=True):

            st.subheader("Hourly Volume Pattern")

            st.caption(
                "Network-wide average traffic volume by hour"
            )

            if "hour" in df.columns:

                hourly = (
                    df.groupby("hour")["traffic_volume"]
                    .mean()
                    .reset_index()
                )

            else:

                hourly = df.copy()

                hourly["hour"] = hourly[
                    "timestamp"
                ].dt.hour

                hourly = (
                    hourly.groupby("hour")[
                        "traffic_volume"
                    ]
                    .mean()
                    .reset_index()
                )

            fig = go.Figure()

            fig.add_trace(

                go.Scatter(

                    x=hourly["hour"],
                    y=hourly["traffic_volume"],

                    mode="lines",

                    line=dict(
                        color="#3B82F6",
                        width=3
                    ),

                    fill="tozeroy",

                    fillcolor="rgba(59,130,246,0.12)",

                    name="Traffic Volume"
                )
            )

            apply_theme(fig)

            fig.update_layout(

                height=380,

                xaxis_title="Hour of Day",

                yaxis_title="Average Vehicles"
            )

            st.plotly_chart(

                fig,

                use_container_width=True,

                config={
                    "displayModeBar": False
                }
            )

    # --------------------------------------------------------
    # CONGESTION SPLIT
    # --------------------------------------------------------

    with col2:

        with st.container(border=True):

            st.subheader("Congestion Split")

            st.caption(
                "Distribution of traffic congestion states"
            )

            if "congestion_level" in df.columns:

                congestion = (
                    df["congestion_level"]
                    .value_counts()
                    .reset_index()
                )

                congestion.columns = [
                    "Congestion",
                    "Count"
                ]

                color_map = {

                    "Free-flow": "#22C55E",
                    "Moderate": "#F59E0B",
                    "Heavy": "#F97316",
                    "Severe": "#EF4444"
                }

                fig = px.pie(

                    congestion,

                    names="Congestion",

                    values="Count",

                    hole=0.65,

                    color="Congestion",

                    color_discrete_map=color_map
                )

                apply_theme(fig)

                fig.update_layout(

                    height=380,

                    showlegend=True
                )

                st.plotly_chart(

                    fig,

                    use_container_width=True,

                    config={
                        "displayModeBar": False
                    }
                )

            else:

                st.info(
                    "Congestion data is not available."
                )

    # --------------------------------------------------------
    # SPEED DISTRIBUTION
    # --------------------------------------------------------

    st.write("")

    with st.container(border=True):

        st.subheader("Speed Distribution")

        st.caption(
            "Distribution of vehicle speeds across the network"
        )

        if "avg_speed" in df.columns:

            fig = px.histogram(

                df,

                x="avg_speed",

                nbins=40,

                labels={
                    "avg_speed": "Speed (km/h)"
                }
            )

            fig.update_traces(
                marker_color="#8B5CF6"
            )

            apply_theme(fig)

            fig.update_layout(
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": False
                }
            )


# ============================================================
# LIVE PREDICTION
# ============================================================

elif active == "Live Prediction":

    page_header(

        "Live Prediction",

        "Predict traffic volume for the selected road segment"
    )

    col1, col2 = st.columns([1, 2])

    with col1:

        with st.container(border=True):

            st.subheader("Prediction Controls")

            roads = sorted(
                df["road_id"]
                .dropna()
                .unique()
            )

            selected_road = st.selectbox(
                "Road Segment",
                roads
            )

            horizon = st.slider(
                "Forecast Horizon",
                min_value=1,
                max_value=12,
                value=1
            )

            run_prediction = st.button(
                "Run Prediction",
                use_container_width=True,
                type="primary"
            )

    with col2:

        with st.container(border=True):

            st.subheader("Prediction Result")

            road_df = (
                df[df["road_id"] == selected_road]
                .sort_values("timestamp")
            )

            latest = road_df.tail(1)

            prediction = model_prediction_frame(
                latest
            )

            if prediction is None:

                st.warning(
                    "No trained XGBoost model found."
                )

                st.info(
                    "Run the model training pipeline first."
                )

            else:

                predicted_value = prediction[0]

                st.metric(

                    "Predicted Traffic Volume",

                    f"{predicted_value:,.0f} vehicles"
                )

                st.divider()

                row = latest.iloc[0]

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(

                        "Current Volume",

                        f"{row.get('traffic_volume', 0):,.0f}"
                    )

                with c2:

                    st.metric(

                        "Average Speed",

                        f"{row.get('avg_speed', 0):.1f} km/h"
                    )

                with c3:

                    st.metric(

                        "Congestion",

                        str(
                            row.get(
                                "congestion_level",
                                "Unknown"
                            )
                        )
                    )

    # --------------------------------------------------------
    # RECENT TREND
    # --------------------------------------------------------

    if "selected_road" in locals():

        with st.container(border=True):

            st.subheader("Recent Traffic Trend")

            recent = road_df.tail(100)

            fig = px.line(

                recent,

                x="timestamp",

                y="traffic_volume"
            )

            fig.update_traces(
                line_color="#3B82F6"
            )

            apply_theme(fig)

            st.plotly_chart(

                fig,

                use_container_width=True,

                config={
                    "displayModeBar": False
                }
            )


# ============================================================
# HISTORICAL TRENDS
# ============================================================

elif active == "Historical Trends":

    page_header(

        "Historical Trends",

        "Explore traffic volume and speed over time"
    )

    roads = ["All"] + sorted(
        df["road_id"]
        .dropna()
        .unique()
        .tolist()
    )

    col1, col2 = st.columns([1, 3])

    with col1:

        selected_road = st.selectbox(
            "Road",
            roads
        )

        metric = st.radio(

            "Metric",

            [
                "Traffic Volume",
                "Average Speed"
            ]
        )

    data = df.copy()

    if selected_road != "All":

        data = data[
            data["road_id"] == selected_road
        ]

    column = (
        "traffic_volume"
        if metric == "Traffic Volume"
        else "avg_speed"
    )

    with col2:

        with st.container(border=True):

            st.subheader(metric)

            fig = px.line(

                data.sort_values("timestamp"),

                x="timestamp",

                y=column
            )

            fig.update_traces(
                line_color="#3B82F6"
            )

            apply_theme(fig)

            st.plotly_chart(

                fig,

                use_container_width=True,

                config={
                    "displayModeBar": False
                }
            )


# ============================================================
# CONGESTION HEATMAP
# ============================================================

elif active == "Congestion Heatmap":

    page_header(

        "Congestion Heatmap",

        "Traffic congestion severity across roads and hours"
    )

    if (
        "congestion_level" not in df.columns
        or "road_id" not in df.columns
    ):

        st.warning(
            "Required congestion data is unavailable."
        )

    else:

        severity_map = {

            "Free-flow": 0,

            "Moderate": 1,

            "Heavy": 2,

            "Severe": 3
        }

        heat_df = df.copy()

        if "hour" not in heat_df.columns:

            heat_df["hour"] = (
                heat_df["timestamp"]
                .dt.hour
            )

        heat_df["severity"] = (
            heat_df["congestion_level"]
            .map(severity_map)
        )

        heatmap = (

            heat_df.groupby(
                ["road_id", "hour"]
            )["severity"]

            .mean()

            .reset_index()

            .pivot(

                index="road_id",

                columns="hour",

                values="severity"
            )
        )

        with st.container(border=True):

            st.subheader(
                "Congestion Severity Grid"
            )

            fig = px.imshow(

                heatmap,

                aspect="auto",

                color_continuous_scale=[

                    "#22C55E",

                    "#F59E0B",

                    "#F97316",

                    "#EF4444"
                ],

                zmin=0,

                zmax=3
            )

            apply_theme(fig)

            fig.update_layout(

                height=max(
                    450,
                    len(heatmap) * 30
                )
            )

            st.plotly_chart(

                fig,

                use_container_width=True
            )


# ============================================================
# ROAD COMPARISON
# ============================================================

elif active == "Road Comparison":

    page_header(

        "Road Comparison",

        "Compare performance metrics across road segments"
    )

    required = [

        "road_id",

        "traffic_volume",

        "avg_speed"
    ]

    if all(
        col in df.columns
        for col in required
    ):

        aggregation = {

            "traffic_volume": "mean",

            "avg_speed": "mean"
        }

        if "travel_time" in df.columns:
            aggregation["travel_time"] = "mean"

        if "occupancy" in df.columns:
            aggregation["occupancy"] = "mean"

        summary = (

            df.groupby("road_id")

            .agg(aggregation)

            .reset_index()
        )

        metric = st.selectbox(

            "Compare by",

            summary.columns[
                1:
            ].tolist()
        )

        with st.container(border=True):

            fig = px.bar(

                summary.sort_values(
                    metric,
                    ascending=True
                ),

                x=metric,

                y="road_id",

                orientation="h"
            )

            fig.update_traces(
                marker_color="#3B82F6"
            )

            apply_theme(fig)

            fig.update_layout(
                height=max(
                    400,
                    len(summary) * 35
                )
            )

            st.plotly_chart(

                fig,

                use_container_width=True,

                config={
                    "displayModeBar": False
                }
            )

        st.subheader("Road Summary")

        st.dataframe(

            summary,

            use_container_width=True,

            hide_index=True
        )


# ============================================================
# WEATHER VS TRAFFIC
# ============================================================

elif active == "Weather vs Traffic":

    page_header(

        "Weather vs Traffic",

        "Relationship between weather conditions and traffic volume"
    )

    sample = df.sample(
        min(5000, len(df)),
        random_state=42
    )

    col1, col2 = st.columns(2)

    with col1:

        if (
            "rainfall" in sample.columns
            and "traffic_volume" in sample.columns
        ):

            with st.container(border=True):

                st.subheader(
                    "Rainfall vs Traffic Volume"
                )

                fig = px.scatter(

                    sample,

                    x="rainfall",

                    y="traffic_volume",

                    opacity=0.6
                )

                apply_theme(fig)

                st.plotly_chart(

                    fig,

                    use_container_width=True,

                    config={
                        "displayModeBar": False
                    }
                )

    with col2:

        if (
            "visibility" in sample.columns
            and "traffic_volume" in sample.columns
        ):

            with st.container(border=True):

                st.subheader(
                    "Visibility vs Traffic Volume"
                )

                fig = px.scatter(

                    sample,

                    x="visibility",

                    y="traffic_volume",

                    opacity=0.6
                )

                apply_theme(fig)

                st.plotly_chart(

                    fig,

                    use_container_width=True,

                    config={
                        "displayModeBar": False
                    }
                )


# ============================================================
# FORECAST VISUALISATION
# ============================================================

elif active == "Forecast Visualisation":

    page_header(

        "Forecast Visualisation",

        "Compare actual traffic volume against model predictions"
    )

    package = load_volume_model()

    if package is None:

        st.warning(
            "No trained model found."
        )

    else:

        roads = sorted(
            df["road_id"]
            .dropna()
            .unique()
        )

        selected_road = st.selectbox(
            "Road Segment",
            roads
        )

        data = (

            df[
                df["road_id"] == selected_road
            ]

            .sort_values("timestamp")

            .tail(300)

            .copy()
        )

        prediction = model_prediction_frame(
            data
        )

        if prediction is not None:

            data["predicted_volume"] = prediction

            actual = data["traffic_volume"]

            predicted = data[
                "predicted_volume"
            ]

            mae = (
                actual - predicted
            ).abs().mean()

            mape = (

                (
                    (
                        actual - predicted
                    ).abs()

                    / actual.replace(
                        0,
                        np.nan
                    )
                )

                .mean()

                * 100
            )

            show_kpis([

                {
                    "label": "Mean Absolute Error",
                    "value": f"{mae:,.1f}"
                },

                {
                    "label": "MAPE",
                    "value": f"{mape:.2f}%"
                },

                {
                    "label": "Data Points",
                    "value": f"{len(data):,}"
                },

                {
                    "label": "Road",
                    "value": str(selected_road)
                }

            ])

            st.write("")

            with st.container(border=True):

                st.subheader(
                    "Actual vs Predicted"
                )

                fig = go.Figure()

                fig.add_trace(

                    go.Scatter(

                        x=data["timestamp"],

                        y=data["traffic_volume"],

                        name="Actual",

                        line=dict(
                            color="#3B82F6",
                            width=2
                        )
                    )
                )

                fig.add_trace(

                    go.Scatter(

                        x=data["timestamp"],

                        y=data["predicted_volume"],

                        name="Predicted",

                        line=dict(
                            color="#F59E0B",

                            width=2,

                            dash="dash"
                        )
                    )
                )

                apply_theme(fig)

                st.plotly_chart(

                    fig,

                    use_container_width=True,

                    config={
                        "displayModeBar": False
                    }
                )


# ============================================================
# PREDICTION CONFIDENCE
# ============================================================

elif active == "Prediction Confidence":

    page_header(

        "Prediction Confidence",

        "Prediction intervals based on model residuals"
    )

    residual_path = (
        MODEL_DIR
        / "xgboost_volume_predictions.csv"
    )

    if not residual_path.exists():

        st.warning(
            "Residual prediction data not found."
        )

    else:

        residual_df = pd.read_csv(
            residual_path
        )

        if "residual" not in residual_df.columns:

            st.warning(
                "Residual column not found."
            )

        else:

            residuals = (
                residual_df["residual"]
                .dropna()
            )

            lower = residuals.quantile(
                0.05
            )

            upper = residuals.quantile(
                0.95
            )

            show_kpis([

                {
                    "label": "Lower Bound",
                    "value": f"{lower:,.0f}"
                },

                {
                    "label": "Upper Bound",
                    "value": f"{upper:,.0f}"
                },

                {
                    "label": "Interval Width",
                    "value": f"{upper - lower:,.0f}"
                },

                {
                    "label": "Confidence Level",
                    "value": "90%"
                }

            ])


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif active == "Model Performance":

    page_header(

        "Model Performance",

        "Performance evaluation of trained FlowCast models"
    )

    scoreboards = [

        (
            MODEL_DIR
            / "regression_scoreboard.csv",

            "Regression Models"
        ),

        (
            MODEL_DIR
            / "congestion_classification_scoreboard.csv",

            "Congestion Classification"
        ),

        (
            MODEL_DIR
            / "accident_risk_classification_scoreboard.csv",

            "Accident Risk Classification"
        ),

        (
            DEEP_MODEL_DIR
            / "lstm_scoreboard.csv",

            "LSTM Model"
        )

    ]

    found = False

    for path, title in scoreboards:

        if path.exists():

            found = True

            with st.container(border=True):

                st.subheader(title)

                scoreboard = pd.read_csv(
                    path
                )

                st.dataframe(

                    scoreboard,

                    use_container_width=True,

                    hide_index=True
                )

    if not found:

        st.info(
            "No model performance files found. "
            "Run the training pipeline first."
        )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

elif active == "Feature Importance":

    page_header(

        "Feature Importance",

        "Features contributing most to traffic volume predictions"
    )

    package = load_volume_model()

    if package is None:

        st.warning(
            "Train the model first."
        )

    else:

        model = package.get("model")
        features = package.get(
            "features",
            []
        )

        estimator = model

        if hasattr(
            model,
            "named_steps"
        ):

            estimator = model.named_steps.get(
                "model",
                model
            )

        if hasattr(
            estimator,
            "feature_importances_"
        ):

            importance = pd.DataFrame({

                "Feature": features,

                "Importance": (
                    estimator
                    .feature_importances_
                )

            })

            importance = (

                importance

                .sort_values(
                    "Importance",
                    ascending=False
                )

                .head(25)
            )

            with st.container(border=True):

                st.subheader(
                    "Top Feature Importances"
                )

                fig = px.bar(

                    importance.sort_values(
                        "Importance"
                    ),

                    x="Importance",

                    y="Feature",

                    orientation="h"
                )

                fig.update_traces(
                    marker_color="#8B5CF6"
                )

                apply_theme(fig)

                st.plotly_chart(

                    fig,

                    use_container_width=True,

                    config={
                        "displayModeBar": False
                    }
                )

        else:

            st.info(
                "Feature importance is not available "
                "for this model."
            )


# ============================================================
# DATA UPLOAD
# ============================================================

elif active == "Data Upload":

    page_header(

        "Data Upload",

        "Upload and validate a new traffic dataset"
    )

    uploaded_file = st.file_uploader(

        "Upload Traffic CSV",

        type=["csv"]
    )

    if uploaded_file:

        uploaded_df = pd.read_csv(
            uploaded_file
        )

        required_columns = {

            "road_id",

            "timestamp",

            "traffic_volume",

            "avg_speed"
        }

        missing = (
            required_columns
            - set(uploaded_df.columns)
        )

        show_kpis([

            {
                "label": "Rows",
                "value": f"{len(uploaded_df):,}"
            },

            {
                "label": "Columns",
                "value": str(
                    uploaded_df.shape[1]
                )
            },

            {
                "label": "Missing Columns",
                "value": str(
                    len(missing)
                )
            }

        ])

        st.write("")

        if missing:

            st.error(

                "Missing required columns: "

                + ", ".join(missing)
            )

        else:

            st.success(
                "Dataset schema looks valid."
            )

        with st.container(border=True):

            st.subheader(
                "Dataset Preview"
            )

            st.dataframe(

                uploaded_df.head(100),

                use_container_width=True,

                hide_index=True
            )


# ============================================================
# REPORTS
# ============================================================

elif active == "Reports":

    page_header(

        "Reports & Insights",

        "Generated analysis and data quality reports"
    )

    report_path = (
        REPORT_DIR
        / "data_quality.md"
    )

    if report_path.exists():

        with st.container(border=True):

            report_content = (
                report_path
                .read_text(
                    encoding="utf-8"
                )
            )

            st.markdown(
                report_content
            )

    else:

        st.info(
            "No data quality report found."
        )

    figures_dir = (
        REPORT_DIR
        / "figures"
    )

    if figures_dir.exists():

        images = sorted(
            figures_dir.glob("*.png")
        )

        if images:

            st.subheader(
                "Generated Figures"
            )

            cols = st.columns(2)

            for i, image in enumerate(images):

                with cols[i % 2]:

                    with st.container(border=True):

                        st.image(
                            str(image),
                            caption=image.stem
                        )