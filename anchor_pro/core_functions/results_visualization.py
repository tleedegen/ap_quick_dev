import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from utils.data_loader import anchor_pro_set_data

def render_anchor_calculation_results():
    """
    Render comprehensive visualization for concrete anchor calculation results.
    Can accept either a CSV file path or a DataFrame directly.
    """

    # Load data
    df = st.session_state['analysis_results_df']

    # Handle case where 'Limit State' might be the index
    if 'Limit State' not in df.columns and df.index.name == 'Limit State':
        df = df.reset_index()
    elif 'Limit State' not in df.columns:
        st.error("'Limit State' column not found in the data")
        return

    # Main title
    st.header("🔧 Concrete Anchor Analysis Results")

    # # Key metrics row
    # col1, col2, col3, col4 = st.columns(4)

    max_utilization = df['Utilization'].max()
    governing_state = df.loc[df['Utilization'].idxmax(), 'Limit State']
    tension_dcr = df[df['Mode'] == 'Tension']['Utilization'].max() if 'Tension' in df['Mode'].values else 0
    shear_dcr = df[df['Mode'] == 'Shear']['Utilization'].max() if 'Shear' in df['Mode'].values else 0

    tab1, tab2, tab3 = st.tabs([
        "📊 Utilization Summary",
        "⚖️ Demand vs Capacity",
        "📋 Data Table"
    ])

    with tab1:
        render_utilization_chart(df)

    with tab2:
        render_demand_capacity_comparison(df)

    with tab3:
        render_data_table(df)

    # Warning messages
    if max_utilization > 1.0:
        st.error(f"⚠️ **CAPACITY EXCEEDED**: {governing_state} has utilization of {max_utilization:.2f}")
    elif max_utilization > 0.9:
        st.warning(f"⚠️ **HIGH UTILIZATION**: {governing_state} is at {max_utilization:.1%} capacity")
    elif max_utilization > 0.8:
        st.info(f"ℹ️ Design utilization is acceptable. Maximum: {max_utilization:.1%}")
    else:
        st.success(f"✅ Design has adequate capacity. Maximum utilization: {max_utilization:.1%}")



def render_utilization_chart(df: pd.DataFrame):
    """Render horizontal bar chart of utilization ratios"""

    # Ensure we have 'Limit State' as a column
    if 'Limit State' not in df.columns and df.index.name == 'Limit State':
        df = df.reset_index()

    # Sort by utilization for better visibility
    df_sorted = df.sort_values('Utilization', ascending=True)

    # Color based on utilization level
    colors = ['red' if x > 1.0 else 'orange' if x > 0.8 else 'green'
              for x in df_sorted['Utilization']]

    fig = go.Figure()

    # Add utilization bars
    fig.add_trace(go.Bar(
        y=df_sorted['Limit State'],
        x=df_sorted['Utilization'],
        orientation='h',
        marker_color=colors,
        text=[f"{x:.2f}" for x in df_sorted['Utilization']],
        textposition='outside',
        name='Utilization',
        hovertemplate='<b>%{y}</b><br>' +
                      'Utilization: %{x:.2f}<br>' +
                      '<extra></extra>'
    ))

    # Add reference lines
    fig.add_vline(x=1.0, line_dash="dash", line_color="red",
                  annotation_text="Capacity Limit", annotation_position="top")
    fig.add_vline(x=0.8, line_dash="dot", line_color="orange",
                  annotation_text="80% Threshold", annotation_position="bottom")

    fig.update_layout(
        title="Limit State Utilization Ratios",
        xaxis_title="Utilization (Demand / Factored Capacity)",
        yaxis_title="",
        height=400,
        xaxis=dict(range=[0, max(1.2, df_sorted['Utilization'].max() * 1.1)]),
        showlegend=False,
        hovermode='y unified'
    )

    st.plotly_chart(fig, use_container_width=True)


def render_demand_capacity_comparison(df: pd.DataFrame):
    """Render grouped bar chart comparing demand vs capacity"""

    # Ensure we have 'Limit State' as a column
    if 'Limit State' not in df.columns and df.index.name == 'Limit State':
        df = df.reset_index()

    # Prepare data for comparison
    comparison_data = df[['Limit State', 'Mode', 'Demand', 'Factored Capacity']].copy()

    # Create subplot with shared y-axis
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Tension Limit States", "Shear Limit States"),
        shared_yaxes=False,
        horizontal_spacing=0.15
    )

    # Separate by mode
    tension_df = comparison_data[comparison_data['Mode'] == 'Tension']
    shear_df = comparison_data[comparison_data['Mode'] == 'Shear']

    # Tension subplot
    if not tension_df.empty:
        fig.add_trace(
            go.Bar(
                name='Demand',
                x=tension_df['Demand'],
                y=tension_df['Limit State'],
                orientation='h',
                marker_color='indianred',
                legendgroup='demand',
                showlegend=True,
                text=[f"{x:.0f}" for x in tension_df['Demand']],
                textposition='outside'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                name='Factored Capacity',
                x=tension_df['Factored Capacity'],
                y=tension_df['Limit State'],
                orientation='h',
                marker_color='lightgreen',
                legendgroup='capacity',
                showlegend=True,
                text=[f"{x:.0f}" for x in tension_df['Factored Capacity']],
                textposition='outside'
            ),
            row=1, col=1
        )

    # Shear subplot
    if not shear_df.empty:
        fig.add_trace(
            go.Bar(
                name='Demand',
                x=shear_df['Demand'],
                y=shear_df['Limit State'],
                orientation='h',
                marker_color='indianred',
                legendgroup='demand',
                showlegend=False,
                text=[f"{x:.0f}" for x in shear_df['Demand']],
                textposition='outside'
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(
                name='Factored Capacity',
                x=shear_df['Factored Capacity'],
                y=shear_df['Limit State'],
                orientation='h',
                marker_color='lightgreen',
                legendgroup='capacity',
                showlegend=False,
                text=[f"{x:.0f}" for x in shear_df['Factored Capacity']],
                textposition='outside'
            ),
            row=1, col=2
        )

    fig.update_layout(
        title="Demand vs Factored Capacity Comparison",
        barmode='group',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='y unified'
    )

    fig.update_xaxes(title_text="Force (lbs)", row=1, col=1)
    fig.update_xaxes(title_text="Force (lbs)", row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)


def render_data_table(df: pd.DataFrame):
    """Render the raw data table with formatting"""

    # Ensure we have 'Limit State' as a column
    if 'Limit State' not in df.columns and df.index.name == 'Limit State':
        df = df.reset_index()

    # Format the dataframe for display
    df_display = df.copy()

    # Apply formatting
    df_display['Demand'] = df_display['Demand'].apply(lambda x: f"{x:,.0f}")
    df_display['Nominal Capacity'] = df_display['Nominal Capacity'].apply(lambda x: f"{x:,.1f}")
    df_display['Reduction Factor'] = df_display['Reduction Factor'].apply(lambda x: f"{x:.2f}")
    df_display['Seismic Factor'] = df_display['Seismic Factor'].apply(lambda x: f"{x:.2f}")
    df_display['Factored Capacity'] = df_display['Factored Capacity'].apply(lambda x: f"{x:,.1f}")
    df_display['Utilization'] = df_display['Utilization'].apply(lambda x: f"{x:.3f}")

    # Display with color coding for utilization
    def highlight_utilization(val):
        try:
            num_val = float(val)
            if num_val > 1.0:
                return 'background-color: #961021'
            elif num_val > 0.8:
                return 'background-color: #84870d'
            else:
                return 'background-color: #0a5618'
        except:
            return ''

    styled_df = df_display.style.map(
        highlight_utilization,
        subset=['Utilization']
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="anchor_analysis_results.csv",
        mime="text/csv"
    )

def render_demand_capacity_ratio_table():
    """Render table of demand/capacity ratios for each saved design (skip default at index 0)."""
    if len(st.session_state['data_column']) <= 1:
        st.info("No saved designs to display")
        return

    original_active_index = st.session_state['active_data_column_index']

    # Collect the Utilization series for each saved design (skip index 0 default)
    ratios = []
    for idx in range(1, len(st.session_state['data_column'])):
        st.session_state['active_data_column_index'] = idx
        anchor_pro_set_data()
        s = st.session_state['analysis_results_df']['Utilization'].copy()
        s.name = f"Design {idx}"   # column label
        ratios.append(s)

    # st.session_state['active_data_column_index'] = original_active_index

    st.write("### 📋 Demand/Capacity Ratios Summary")

    # Combine into a single DataFrame with designs as columns
    ratios_df = pd.concat(ratios, axis=1)

    # Styling helper (use numeric value, not strings)
    def highlight_utilization(val):
        try:
            if val > 1.0:
                return 'background-color: #961021; color: white;'
            elif val > 0.8:
                return 'background-color: #84870d; color: white;'
            else:
                return 'background-color: #0a5618; color: white;'
        except Exception:
            return ''

    # Format to 3 decimals *and* apply per-cell colors
    styled = (
        ratios_df.style.format("{:.3f}").map(highlight_utilization)
    )

    # IMPORTANT: pass the Styler to Streamlit (not the raw DataFrame)
    st.dataframe(styled, use_container_width=True, height=500)




