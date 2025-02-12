import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def streamlit_prediction_stats(df_raw, route, threshold):
    # Filter the dataframe for the given route
    df_route = df_raw[df_raw['route_code'] == route].copy()

    # Dynamically classify predictions based on the slider threshold
    df_route['Prediction'] = df_route['predicted_prob_class_1'].apply(
        lambda x: 'On-Time' if x < threshold else 'Delayed'
    )

    # Recalculate TP, FP, TN, FN based on the new threshold
    df_route['TP_dynamic'] = (df_route['true_label'] == 1) & (df_route['predicted_prob_class_1'] >= threshold)
    df_route['FP_dynamic'] = (df_route['true_label'] == 0) & (df_route['predicted_prob_class_1'] >= threshold)
    df_route['TN_dynamic'] = (df_route['true_label'] == 0) & (df_route['predicted_prob_class_1'] < threshold)
    df_route['FN_dynamic'] = (df_route['true_label'] == 1) & (df_route['predicted_prob_class_1'] < threshold)

    # Sum up dynamic TP, FP, TN, FN
    TP = df_route['TP_dynamic'].sum()
    FP = df_route['FP_dynamic'].sum()
    TN = df_route['TN_dynamic'].sum()
    FN = df_route['FN_dynamic'].sum()

    # Calculate precision and recall
    precision = df_route['TP'].sum() / (df_route['TP'].sum() + df_route['FP'].sum())
    recall = df_route['TP'].sum() / (df_route['TP'].sum() + df_route['FN'].sum())

    # Calculate accuracy dynamically
        # Create a dataframe with the prediction stats for the given route
    df_route_stats = pd.DataFrame({
        'route': [route],
        'total_flights': [df_route.shape[0]],
        'total_delayed': [df_route['true_label'].sum()],
        'total_ontime': [df_route['true_label'].count() - df_route['true_label'].sum()],
        'percent_delayed': [df_route['true_label'].sum() / df_route['true_label'].count()],
        'total_FP': [df_route['FP'].sum()],
        'total_FN': [df_route['FN'].sum()],
        'total_TP': [df_route['TP'].sum()],
        'total_TN': [df_route['TN'].sum()],
        'FP_rate': [df_route['FP'].sum() / df_route['true_label'].count()],
        'FN_rate': [df_route['FN'].sum() / df_route['true_label'].count()],
        'TP_rate': [df_route['TP'].sum() / df_route['true_label'].count()],
        'TN_rate': [df_route['TN'].sum() / df_route['true_label'].count()],
        'accuracy': [(df_route['TP'].sum() + df_route['TN'].sum()) / df_route['true_label'].count()],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [2 * (precision * recall) / (precision + recall)]
    })

    # Plotly Histogram
    fig_hist = px.histogram(df_route, 
                             x='predicted_prob_class_1', 
                             nbins=20, 
                             color='Prediction',
                             color_discrete_map={'On-Time': 'cornflowerblue', 'Delayed': 'crimson'})
    fig_hist.update_xaxes(title_text='Probability of Delay', range=[0, 1])
    fig_hist.update_yaxes(title_text=f'No. of Predictions Made')

    # Update traces (Bin Opacity & Borders)
    fig_hist.update_traces(marker=dict(
        line=dict(color='black', width=1.5),
        opacity=0.5
    ))

    # Add a dynamic vertical line at the selected threshold
    fig_hist.add_shape(
        type="line",
        x0=threshold, y0=0, x1=threshold, y1=0.9,
        line=dict(color="red", width=5, dash="dash"),
        xref="x", yref="paper"
    )
    fig_hist.add_annotation(
        x=threshold, y=0.97,
        text=f"Threshold = {threshold:.1f}",
        showarrow=False,
        yshift=10,
        font=dict(color="red"),
        xref="x",
        yref="paper"
    )

    # Adjust Figure Layout
    fig_hist.update_layout(
        autosize=False,
        width=600,
        height=400,
        margin=dict(l=20, r=20, t=0, b=20),
        legend=dict(
            # Place the legend inside the plot area
            x=1.20,        # Use 0.95 instead of 1 to avoid expanding the figure
            xanchor="right",
            y=0.18,        # Lower the legend by setting y to 0.85 (adjust as needed)
            yanchor="top"
        )
    )

    # ---------------------------------------------------
    # DYNAMIC PIE CHARTS (Legends moved to the left)
    # ---------------------------------------------------
    
    # Pie chart for Delayed Predictions
    fig_pie_delayed = go.Figure()
    fig_pie_delayed.add_trace(go.Pie(labels=['Correctly Predicted (TP)', 'Falsely Predicted (FP)'], 
                                     values=[TP, FP], 
                                     name='Delayed', 
                                     marker=dict(colors=['crimson', 'lightcoral'])))
    fig_pie_delayed.update_traces(hole=.4, hoverinfo="label+percent+name", textinfo="percent", 
                                  marker=dict(line=dict(color='black', width=2)))
    fig_pie_delayed.update_layout(
        title_text='Departure Delay', 
        autosize=False, width=400, height=241, 
        margin=dict(l=0, r=0, t=40, b=20),
        legend=dict(orientation="h")  # Legend moved to the left
    )

    # Pie chart for On-Time Predictions
    fig_pie_ontime = go.Figure()
    fig_pie_ontime.add_trace(go.Pie(labels=['Correctly Predicted (TN)', 'Falsely Predicted (FN)'], 
                                    values=[TN, FN], 
                                    name='On-Time', 
                                    marker=dict(colors=['cornflowerblue', 'lightblue'])))
    fig_pie_ontime.update_traces(hole=.4, hoverinfo="label+percent+name", textinfo="percent", 
                                 marker=dict(line=dict(color='black', width=2)))
    fig_pie_ontime.update_layout(
        title_text='Departure On-Time', 
        autosize=False, width=400, height=241, 
        margin=dict(l=0, r=0, t=40, b=20),
        legend=dict(orientation="h")  # Legend moved to the left
    )

    # Return Updated Data & Figures
    return df_route_stats, fig_hist, fig_pie_delayed, fig_pie_ontime
