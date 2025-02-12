import plotly.graph_objects as go
import pandas as pd

def departure_delay_prog_for_route_group(df_raw, icao_dep,regionalize=False,routes=[], operators=[], aircraft_types=[]):
    # TODO: Consider regional option, how to visualize. and how to integrate into streamlit
    if regionalize:
        # Extract all rows where the 'origin_region' is the same as the 'origin_region' corresponding to the input 'icao_dep'
        df_filtered = df_raw[df_raw['origin_region'] == df_raw[df_raw['origin.code_icao'] == icao_dep]['origin_region'].iloc[0]]
    else :
        # Extract all rows where the 'origin.code_icao' is equal to the input 'icao_dep'
        df_filtered = df_raw[df_raw['origin.code_icao'] == icao_dep]

        # If a list of routes is provided, filter the data further by the routes in the list
        if len(routes) > 0 :
            df_filtered = df_filtered[df_filtered['ICAO_route'].isin(routes)]

        # If a list of operators is provided, filter the data further by the operators in the list
        if len(operators) > 0 :
            df_filtered = df_filtered[df_filtered['operator_icao'].isin(operators)]

        # If a list of aircraft types is provided, filter the data further by the aircraft types in the list
        if len(aircraft_types) > 0 :
            df_filtered = df_filtered[df_filtered['aircraft_type'].isin(aircraft_types)]

    # Group by 'year_week' and 'departure_delay_binary', count the occurrences, and pivot the table to have departure_delay_binary as columns and year_week as index
    df_grouped_weekly = df_filtered.groupby(['year_week', 'departure_delay_binary']).size().reset_index(name='count')
    df_pivot_weekly = df_grouped_weekly.pivot(index='year_week', columns='departure_delay_binary', values='count').fillna(0)
    df_pivot_weekly = df_pivot_weekly[['delayed', 'on_time']] 
    df_pivot_weekly['total'] = df_pivot_weekly.sum(axis=1)

    df_pivot_weekly.index = pd.to_datetime(df_pivot_weekly.index)

    # Initialize Figure
    fig = go.Figure()

    # Plot moving averages for 'total' and 'delayed' columns as a line plot versus the 'year_week' index in plotly with a 4 week window
    fig.add_trace(go.Scatter(x=df_pivot_weekly.index, y=df_pivot_weekly['total'].rolling(window=4).mean(), mode='lines', line=dict(color='blue'), name='All Departures'))
    fig.add_trace(go.Scatter(x=df_pivot_weekly.index, y=df_pivot_weekly['delayed'].rolling(window=4).mean(), mode='lines', line=dict(color='red'), name='Delayed Departures'))
    
    # Plot 'total' and 'delayed' columns as a line plot versus the 'year_week' index in plotly
    fig.add_trace(go.Scatter(x=df_pivot_weekly.index, y=df_pivot_weekly['total'], mode='lines', line=dict(color='blue'), opacity=0.15, name='All Departures',showlegend=False))
    fig.add_trace(go.Scatter(x=df_pivot_weekly.index, y=df_pivot_weekly['delayed'], mode='lines', line=dict(color='red'), opacity=0.15, name='Delayed Departures',showlegend=False))

    # Shade in timeframtes of interest with conditionality on 'origin_region' 
    seasonality = ''
    # 'Africa & Middle East': 'Ramadan' -> sandybrown, 0.3 
    if df_filtered['origin_region'].iloc[0] == 'Africa & Middle East':
        seasonality = 'Ramadan Periods'
        # Ramadan 2022
        fig.add_vrect(x0="2022-04-02", x1="2022-05-01", fillcolor="sandybrown", opacity=0.3, layer="below", line_width=0)
        fig.add_annotation(x="2022-04-17", y=0, text="Ramadan 2022", showarrow=False, font=dict(size=10, color='black'))
        # Ramadan 2023
        fig.add_vrect(x0="2023-03-23", x1="2023-04-20", fillcolor="sandybrown", opacity=0.3, layer="below", line_width=0)
        fig.add_annotation(x="2023-04-10", y=0, text="Ramadan 2023", showarrow=False, font=dict(size=10, color='black'))
        # Ramadan 2024
        fig.add_vrect(x0="2024-03-10", x1="2024-04-09", fillcolor="sandybrown", opacity=0.3, layer="below", line_width=0)
        fig.add_annotation(x="2024-03-25", y=0, text="Ramadan 2024", showarrow=False, font=dict(size=10, color='black'))
        # Post Covid Rebounce from start until August 15 2022, add label
        fig.add_vrect(x0="2022-01-31", x1="2022-08-15", fillcolor="wheat", opacity=0.3, layer="below", line_width=0)
        fig.add_annotation(x="2022-08-15", y=0, text="Post-Covid Recovery", showarrow=False, font=dict(size=10, color='black'))

    # 'Asia Pacific': 'Monsoon' -> mediumseagreen, 0.2
    elif df_filtered['origin_region'].iloc[0] == 'Asia Pacific':
        seasonality = 'Monsoon seasons'
        # Monsoon 2022
        fig.add_vrect(x0="2022-05-15", x1="2022-10-15", fillcolor="mediumseagreen", opacity=0.3, layer="below", line_width=0)
        fig.add_annotation(x="2022-09-30", y=0, text="Monsoon 2022", showarrow=False, font=dict(size=10, color='black'))
        # Monsoon 2023
        fig.add_vrect(x0="2023-05-15", x1="2023-10-15", fillcolor="mediumseagreen", opacity=0.3, layer="below", line_width=0)
        fig.add_annotation(x="2023-07-31", y=0, text="Monsoon 2023", showarrow=False, font=dict(size=10, color='black'))
        # Monsoon 2024
        fig.add_vrect(x0="2024-05-15", x1="2024-10-15", fillcolor="mediumseagreen", opacity=0.3, layer="below", line_width=0)
        fig.add_annotation(x="2024-07-31", y=0, text="Monsoon 2024", showarrow=False, font=dict(size=10, color='black'))
        # Post Covid Rebounce from start until August 15 2022, add label
        fig.add_vrect(x0="2022-01-31", x1="2022-08-15", fillcolor="wheat", opacity=0.3, layer="below", line_width=0)
        fig.add_annotation(x="2022-05-15", y=0, text="Post-Covid Recovery", showarrow=False, font=dict(size=10, color='black'))


    # All the rest: 'Winter' -> cornflowerblue, 0.3
    else :
        seasonality = 'Winters'
        # December 2022-2023
        fig.add_vrect(x0="2022-12-01", x1="2023-01-31", fillcolor="cornflowerblue", opacity=0.3, layer="below", line_width=0)
        fig.add_annotation(x="2022-12-31", y=0, text="Winter 22-23", showarrow=False, font=dict(size=10, color='black'))
        # December 2023-2024
        fig.add_vrect(x0="2023-12-01", x1="2024-01-31", fillcolor="cornflowerblue", opacity=0.3, layer="below", line_width=0)
        fig.add_annotation(x="2023-12-31", y=0, text="Winter 23-24", showarrow=False, font=dict(size=10, color='black'))
        # December 2024-2025
        fig.add_vrect(x0="2024-12-01", x1=df_pivot_weekly.index[-1].strftime('%Y-%m-%d'), fillcolor="cornflowerblue", opacity=0.3, layer="below", line_width=0)
        fig.add_annotation(x="2024-11-20", y=0, text="Winter 24-25", showarrow=False, font=dict(size=10, color='black'))
        # Post Covid Rebounce from start until August 15 2022, add label
        fig.add_vrect(x0="2022-01-31", x1="2022-08-15", fillcolor="wheat", opacity=0.3, layer="below", line_width=0)
        fig.add_annotation(x="2022-05-15", y=0, text="Post-Covid Recovery", showarrow=False, font=dict(size=10, color='black'))
        
    # Configure X-Axis and Ticks
    tickvals = df_pivot_weekly.index.to_list()
    major_ticks = []
    for i, date in enumerate(tickvals):
        if i == 0 :
            major_ticks.append(date)
        elif date.month == 1 and date.week == 1:
            major_ticks.append(date)
        elif date.month == 7 and date.week == 1:
            major_ticks.append(date)

    major_ticktext = [date.strftime('%b\n%Y') for date in major_ticks]

    fig.update_xaxes(ticks= "outside",
                 ticklabelmode= "period", 
                 tickcolor= "black", 
                 ticklen=10, 
                 tickvals=major_ticks,
                 ticktext=major_ticktext,
                 tickangle=15,
                 tickmode='array',
                 tickson='boundaries',
                 minor=dict(
                     ticklen=4,
                     dtick="M3",
                     tick0="2022-04-01",
                     griddash='dot',
                     gridcolor='white'
                 )
    )
    
    # Figure Title Metaparameters
    if len(routes) == 0:
        routes = 'All'
    elif len(routes) > 5:
        routes = routes[0:5]
    
    if len(operators) == 0:
        operators = 'All'
    elif len(operators) > 5:
        operators = operators[0:5]

    if len(aircraft_types) == 0:
        aircraft_types = 'All'
    elif len(aircraft_types) > 5:
        aircraft_types = aircraft_types[0:5]
    
    # Region Dependent Parameters
    if regionalize:
        region_name = df_filtered['origin_region'].iloc[0]
        title_text = f"Weekly Departures in Dataset for {region_name} region<br><sup>Routes: {routes} -- Operators: {operators} -- Aircraft: {aircraft_types}</sup>"
    else:
        airport_name = df_filtered['origin_airport_name'].iloc[0]
        airport_location = df_filtered['origin_airport_location'].iloc[0]
        title_text = f"Weekly Departures in Dataset: {airport_name} ({icao_dep}) - {airport_location}<br><sup>Routes: {routes} -- Operators: {operators} -- Aircraft: {aircraft_types}</sup>"

    # Update Layout
    fig.update_layout(title={'text': title_text,
                             'y':0.9,
                             'x':0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                        xaxis_title='Date',
                        yaxis_title='Number of Departures per Week',
                        legend_title='4-week Moving Average',
                        height=600,
                        width=1200)

    # Show Legend
    fig.update_layout(legend=dict(x=0.85,
                                  y=-0.2,
                                  traceorder='normal',
                                  bgcolor='rgba(255, 255, 255, 0.5)',
                                  bordercolor='black',
                                  borderwidth=1
                              ))

    # Show Plot and return
    return fig
