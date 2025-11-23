from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Optional, Dict, Any, List, Mapping
from core_functions.results_visualization import render_anchor_calculation_results, render_demand_capacity_ratio_table
from core_functions.design_parameters import DesignParameters
from utils.exceptions import DesignNameError
from utils.data_loader import anchor_pro_set_data

def render_visualizations():
    """Render all visualizations for anchor data"""
    # if anchor_data is None or anchor_data.empty:
    #     st.info("No anchor data to visualize. Add anchors in the editor.")
    #     return

    if len(st.session_state['data_column']) > 1:
        active_index_selectbox = st.selectbox(label='Select Design to Visualize',
                                    options=(range(1, len(st.session_state['data_column']))) if len(st.session_state['data_column']) > 1 else [0],
                                    key='visual_index',
                                    format_func=lambda x: f"Design {x}" )
        active_index = active_index_selectbox

        if st.session_state['active_data_column_index'] != active_index:
            st.session_state['active_data_column_index'] = active_index

        anchor_pro_set_data(active_index)
        active_geometry_forces = st.session_state['data_column'][active_index]['anchor_geometry_forces']

        col1, col2 = st.columns(2)

        with col1:
            render_anchor_layout(active_geometry_forces, active_index)
        with col2:
            render_anchor_calculation_results()
    else:
        st.info("No saved designs to visualize. Record a design to see visualizations.")

def render_anchor_layout(df: pd.DataFrame, data_column_index: int):
    """Render 2D anchor layout visualization with bounding box"""
    fig = go.Figure()

    # Add anchor points
    fig.add_trace(go.Scatter(
        x=df['X'],
        y=df['Y'],
        mode='markers+text',
        marker=dict(
            size=15,
            color='blue',
            symbol='circle',
            line=dict(width=2, color='darkblue')
        ),
        text=[f"A{i+1}" for i in range(len(df))],
        textposition="top right",
        name="Anchors",
        hovertemplate='Anchor %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
    ))

    # Calculate centroid
    centroid_x = df['X'].mean()
    centroid_y = df['Y'].mean()

    # Add centroid if multiple anchors
    if len(df) > 1:
        fig.add_trace(go.Scatter(
            x=[centroid_x],
            y=[centroid_y],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                symbol='x',
                line=dict(width=2, color='darkred')
            ),
            name="Centroid",
            hovertemplate='Centroid<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))

    # Add bounding box (baseplate) from session state
    if 'data_column' in st.session_state and st.session_state['data_column']:
        Bx = st.session_state['data_column'][data_column_index].get('Bx', 24.0)  # Default to 24 if not found
        By = st.session_state['data_column'][data_column_index].get('By', 24.0)  # Default to 24 if not found

        # Calculate box corners centered on centroid
        half_width = Bx / 2
        half_height = By / 2

        box_x = [
            centroid_x - half_width,  # Bottom-left
            centroid_x + half_width,  # Bottom-right
            centroid_x + half_width,  # Top-right
            centroid_x - half_width,  # Top-left
            centroid_x - half_width   # Close the box
        ]

        box_y = [
            centroid_y - half_height,  # Bottom-left
            centroid_y - half_height,  # Bottom-right
            centroid_y + half_height,  # Top-right
            centroid_y + half_height,  # Top-left
            centroid_y - half_height   # Close the box
        ]

        # Add the bounding box
        fig.add_trace(go.Scatter(
            x=box_x,
            y=box_y,
            mode='lines',
            line=dict(
                color='green',
                width=2,
                dash='solid'
            ),
            name=f"Baseplate ({Bx}\" × {By}\")",
            hovertemplate='Baseplate Boundary<br>Width: ' + f'{Bx}' + ' in<br>Height: ' + f'{By}' + ' in<extra></extra>'
        ))

        # Add corner markers for clarity
        fig.add_trace(go.Scatter(
            x=box_x[:-1],  # Exclude the closing point
            y=box_y[:-1],
            mode='markers',
            marker=dict(
                size=6,
                color='green',
                symbol='square'
            ),
            name="Baseplate Corners",
            showlegend=False,
            hovertemplate='Corner<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))

    # Calculate appropriate axis range to show everything
    if 'data_column' in st.session_state and st.session_state['data_column']:
        Bx = st.session_state['data_column'][data_column_index].get('Bx', 24.0)
        By = st.session_state['data_column'][data_column_index].get('By', 24.0)

        # Get min/max values for proper scaling
        all_x = list(df['X']) + [centroid_x - Bx/2, centroid_x + Bx/2]
        all_y = list(df['Y']) + [centroid_y - By/2, centroid_y + By/2]

        x_range = [min(all_x) - 2, max(all_x) + 2]
        y_range = [min(all_y) - 2, max(all_y) + 2]

        # Make the range square (equal aspect ratio)
        x_span = x_range[1] - x_range[0]
        y_span = y_range[1] - y_range[0]
        if x_span > y_span:
            diff = (x_span - y_span) / 2
            y_range[0] -= diff
            y_range[1] += diff
        else:
            diff = (y_span - x_span) / 2
            x_range[0] -= diff
            x_range[1] += diff
    else:
        x_range = None
        y_range = None

    # Update layout
    fig.update_layout(
        title="Anchor Layout with Baseplate",
        xaxis_title="X (in)",
        yaxis_title="Y (in)",
        height=800,
        showlegend=True,
        hovermode='closest',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            range=x_range
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='black',
            scaleanchor="x",  # This ensures equal aspect ratio
            scaleratio=1,      # 1:1 ratio
            range=y_range
        )
    )

    # Add annotations for dimensions
    if 'data_column' in st.session_state and st.session_state['data_column']:
        Bx = st.session_state['data_column'][data_column_index].get('Bx', 24.0)
        By = st.session_state['data_column'][data_column_index].get('By', 24.0)

        # Add dimension annotations
        fig.add_annotation(
            x=centroid_x,
            y=centroid_y - By/2 - 1,
            text=f"{Bx}\"",
            showarrow=False,
            font=dict(size=12, color="green"),
            yshift=-10
        )

        fig.add_annotation(
            x=centroid_x - Bx/2 - 1,
            y=centroid_y,
            text=f"{By}\"",
            showarrow=False,
            font=dict(size=12, color="green"),
            xshift=-10,
            textangle=-90
        )

    st.plotly_chart(fig, use_container_width=True)

def render_force_distribution(df: pd.DataFrame):
    """Render force distribution among anchors"""
    if not all(col in df.columns for col in ['Vx', 'Vy', 'N']):
        st.warning("Missing force data for visualization")
        return

    # Create subplots for different force components
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart for forces
        force_data = pd.DataFrame({
            'Anchor': [f"A{i+1}" for i in range(len(df))],
            'Vx': df['Vx'],
            'Vy': df['Vy'],
            'N': df['N']
        })

        fig_bar = px.bar(
            force_data.melt(id_vars=['Anchor'], var_name='Force', value_name='Value'),
            x='Anchor',
            y='Value',
            color='Force',
            title='Force Distribution by Anchor',
            barmode='group'
        )

        fig_bar.update_layout(
            yaxis_title="Force (lbs)",
            xaxis_title="Anchor ID"
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        # Pie chart for tension distribution
        if df['N'].sum() > 0:
            fig_pie = px.pie(
                values=df['N'],
                names=[f"A{i+1}" for i in range(len(df))],
                title='Tension Force Distribution'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No tension forces to display")

def render_analysis_results(df: pd.DataFrame):
    """Render analysis results and calculations"""

    # Calculate resultant forces
    total_vx = df['Vx'].sum()
    total_vy = df['Vy'].sum()
    total_n = df['N'].sum()
    resultant_v = np.sqrt(total_vx**2 + total_vy**2)

    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Vx", f"{total_vx:.1f} lbs")
    with col2:
        st.metric("Total Vy", f"{total_vy:.1f} lbs")
    with col3:
        st.metric("Total N", f"{total_n:.1f} lbs")
    with col4:
        st.metric("Resultant V", f"{resultant_v:.1f} lbs")

    # Mathematical formulas display
    if st.checkbox("Show Calculations"):
        render_calculation_display(df)


def render_calculation_display(df: pd.DataFrame):
    """Display mathematical calculations and formulas"""

    st.markdown("### Force Equilibrium Check")

    # Example calculations
    st.markdown(r"""
    **Resultant Shear Force:**
    $$V_{\text{resultant}} = \sqrt{V_x^2 + V_y^2}$$
    """)

    # Show actual calculation
    total_vx = df['Vx'].sum()
    total_vy = df['Vy'].sum()
    resultant = np.sqrt(total_vx**2 + total_vy**2)

    # st.markdown(f"""
    # $$V_{{\\text{{resultant}}}} = \\sqrt{{{total_vx:.1f}}^2 + {total_vy:.1f}^2}} = {resultant:.1f} \\text{{ lbs}}$$
    # """)

    st.markdown(f"""
    $$
    V_{{\\text{{resultant}}}} = \\sqrt{{{total_vx:.1f}^2 + {total_vy:.1f}^2}} = {resultant:.1f}\\ \\text{{lbs}}
    $$
    """)


    # Add more calculations as needed
    if len(df) > 1:
        st.markdown("### Moment Equilibrium")
        centroid_x = df['X'].mean()
        centroid_y = df['Y'].mean()

        moments_x = ((df['Y'] - centroid_y) * df['N']).sum()
        moments_y = ((df['X'] - centroid_x) * df['N']).sum()

        st.markdown(f"""
        **Moments about centroid:**
        - $M_x = {moments_x:.1f}$ lb-in
        - $M_y = {moments_y:.1f}$ lb-in
        """)


def render_project_designs_table():
    """Render the project designs summary table with design switching functionality"""
    with st.expander('Project Designs Summary', expanded=True):

        designs: list = st.session_state['data_column']
        designs_trimmed: list = designs[1:]  # Exclude the default design at index 0
        if 'design_names' in st.session_state:
            names = st.session_state['design_names']
        else:
            names = []

        if len(designs_trimmed) < 1:
            st.info("No designs saved yet. Use 'Record Data' to save designs.")
            return

        df = pd.DataFrame(designs_trimmed)

        # Add design labels as row index
        if len(names) == len(designs_trimmed):
            name_index = st.session_state['design_names']
            df.index = pd.Index(name_index)
        else:
            raise DesignNameError(f"Mismatch between number of design names and saved designs. Names: {len(names)}, Designs: {len(designs_trimmed)}")

        # Display the dataframe transposed (designs as columns, parameters as rows)
        st.dataframe(
            data=df.T,
            height=1000,
            use_container_width=True
        )

        st.markdown('**Delete Designs**')

        # Select which design to modify (index shifted by +1 because 0 is reserved)
        selected_design = st.selectbox(
            "Select a design to modify:",
            options=list(range(len(designs_trimmed))),
            format_func=lambda i: names[i]
        )

        if st.button("Delete Selected Design"):
            del designs[selected_design + 1]  # +1 to account for default design at index 0
            del names[selected_design]
            st.session_state['design_names'] = names
            st.session_state['data_column'] = designs
            st.session_state['active_data_column_index'] = 1  # Reset active index to 1
            st.success(f"Design {selected_design} deleted.")

            st.rerun()  # Refresh to show changes



def clear_designs():
    """Clears designs and resets to default data column"""
    default_data_column: list = []
    design_params = DesignParameters()

    default_series = pd.Series(design_params.combined_dict)
    default_data_column.insert(0, default_series)
    return default_data_column

# visualizations.py
# Update for render_dynamic_design_table() — split first table into multiple labeled tables


def _titleize(name: str) -> str:
    """Turn snake_case / camelCase-ish into Title Case with spaces."""
    if not isinstance(name, str):
        return str(name)
    s = name.replace("_", " ")
    out = []
    buf = s[:1]
    for ch in s[1:]:
        if ch.isupper() and (buf and not buf[-1].isspace() and not buf[-1].isupper()):
            out.append(buf)
            buf = ch
        else:
            buf += ch
    out.append(buf)
    titled = " ".join(out).strip()
    titled = " ".join(w.capitalize() if w.isupper() else w.title() for w in titled.split())
    return titled


def _normalize_scalar_table(obj: Mapping[str, Any]) -> pd.DataFrame:
    """Convert a mapping of scalars into a vertical two-column DataFrame with nice labels.
    Keeps a hidden raw key column for categorization.
    """
    rows = []
    for k, v in obj.items():
        if isinstance(v, (dict, list, tuple, set, pd.DataFrame, pd.Series)):
            # skip nested – caller will handle separately
            continue
        rows.append((str(k), _titleize(k), v))
    df = pd.DataFrame(rows, columns=["_raw_key", "Parameter", "Value"]).set_index("Parameter")
    return df


def _as_dataframe(value: Any) -> Optional[pd.DataFrame]:
    """Best-effort conversion of common nested structures to a readable DataFrame."""
    if value is None:
        return None
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame(name="Value")
    if isinstance(value, Mapping):
        if all(not isinstance(v, Mapping) for v in value.values()):
            return pd.DataFrame({"Value": value}).rename(index=_titleize)
        return pd.DataFrame(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return pd.DataFrame()
        if all(hasattr(x, "keys") for x in value):
            return pd.DataFrame(value)
        return pd.DataFrame({"Value": list(value)})
    return pd.DataFrame({"Value": [value]})


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(map(str, tup)).strip() for tup in df.columns]
    df.columns = [_titleize(str(c)) for c in df.columns]
    return df


# ---------- categorization for the first (summary) table ----------

KEYWORD_CATEGORIES = [
    ("anchor", "Anchor"),
    ("geometry", "Geometry"),
    ("force", "Forces"),
    ("load", "Loads"),
    ("material", "Materials"),
    ("concrete", "Concrete"),
    ("steel", "Steel"),
    ("substrate", "Substrate"),
    ("factor", "Factors"),
    ("safety", "Safety"),
    ("design", "Design"),
]


def _categorize_param(raw_key: str) -> str:
    key = raw_key.lower().replace("_", " ")
    for kw, label in KEYWORD_CATEGORIES:
        if kw in key:
            return label
    return "Other"


def _split_scalar_table(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    buckets: dict[str, list[tuple[str, Any]]] = {}
    for raw_key, pretty_label, value in zip(df["_raw_key"].values, df.index.tolist(), df["Value"].values):
        cat = _categorize_param(raw_key)
        buckets.setdefault(cat, []).append((pretty_label, value))
    out: dict[str, pd.DataFrame] = {}
    ordered = sorted([c for c in buckets if c != "Other"]) + (["Other"] if "Other" in buckets else [])
    for cat in ordered:
        rows = buckets[cat]
        cdf = pd.DataFrame(rows, columns=["Parameter", "Value"]).set_index("Parameter")
        out[cat] = cdf
    return out


def render_dynamic_design_table() -> None:
    """Render the dynamic design table for preview"""
    with st.expander('Editor Preview', expanded=False):
        with st.container():
            data_column: pd.Series = st.session_state['data_column'][st.session_state.dynamic_data_column_index].copy(deep=True)

            substrate_params_table = data_column[['fc', 'cracked_concrete', 'weight_classification_base',
                                                'poisson', 'grouted', 'profile',
                                                't_slab', 'cx_neg', 'cx_pos', 'cy_neg', 'cy_pos']]
            st.dataframe(substrate_params_table, use_container_width=True)

            anchor_table = data_column[['specified_product', 'seismic', 'phi_override']]
            st.dataframe(anchor_table, use_container_width=True)

            with st.expander(f"{data_column['specified_product']} Specs", expanded=False):
                anchor_parameters = data_column['anchor_parameters']
                specified_anchor_parameters_index = anchor_parameters.index[anchor_parameters['anchor_id'] == data_column['specified_product']][0]
                specified_anchor_parameters = anchor_parameters.iloc[specified_anchor_parameters_index]
                st.dataframe(specified_anchor_parameters)


