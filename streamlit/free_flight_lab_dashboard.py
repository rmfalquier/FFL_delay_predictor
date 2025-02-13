import streamlit as st

st.set_page_config(page_icon="airplane", page_title="Free Flight Lab", layout="wide")

import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import colorsys
from matplotlib import colormaps
import matplotlib.colors as mcolors
import streamlit_depdelay_graph as rene
import streamlit_stats_graphs as rene2
from streamlit_option_menu import option_menu
import streamlit_metar_parse as smp
import plotly.express as px
from copy import deepcopy

# Define the navigation menu
selected = option_menu(
    None, 
    ["Free Flight Lab", "Delay Map", "Prediction", "Contact"], 
    icons=["airplane", "map", "graph-up", "person-circle"], 
    menu_icon="cast", 
    default_index=0, 
    orientation="horizontal"
)

# Load the data
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df_raw = load_data(path='streamlit/streamlit_data/streamlit_map_1_depdelays_per_airport.csv')
df = deepcopy(df_raw)

df_names_routes_raw = load_data(path='streamlit/streamlit_data/streamlit_map_1_names_routes.csv')
df_names_routes = deepcopy(df_names_routes_raw)

df_metar_raw = load_data(path='streamlit/streamlit_data/streamlit_map_2_ml_results.csv')
df_metar = deepcopy(df_metar_raw)

df_route_cities_raw = load_data(path='streamlit/streamlit_data/streamlit_map_2_route_cities.csv')
df_route_cities = deepcopy(df_route_cities_raw)

# Rename columns for better readability
df = df.drop(columns='Unnamed: 0')
df.rename(columns={'delayed_count':'Delayed Flights','num_flights':'Number of Flights','delay_percentage':'Percentage of Departure Delays'}, inplace=True)


# ---------- First Page: Free Flight Lab ----------
if selected == "Free Flight Lab":
    st.title("Predicting weather-related flight delays")
    st.write("""
    - We analyze delays at origin airports when departures are delayed by more than 15 minutes beyond the scheduled time. 
    - Our study leverages flight data from Flightradar24 and Flightaware as well as weather data from AVWX.
    - The dataset includes 220,000 flights across 64 routes over three years, with over 1,000 flights considered per route.
    - To predict delays, we utilize a Random Forest algorithm. 
    - These predictions are designed to help professionals in the aviation industry take proactive measures to address potential flight delays.""")
    # Load the data (You may need to adjust this path to your file)
    @st.cache_data
    def load_data():
        return pd.read_csv("streamlit/streamlit_data/streamlit_map_2_ml_results.csv")

    # Load data
    df_raw = load_data()

    # --- First plot: Delayed flights by route ---

    # Group by route_code and calculate the number of delayed flights and total flights
    df_delayed_route = df_raw.groupby('route_code').agg(
        delayed_count=('true_label', lambda x: (x == 1).sum()),
        total_flights=('true_label', 'size')  # Total number of flights
    )

    # Calculate the percentage of delayed flights
    df_delayed_route['percentage'] = round(
        df_delayed_route['delayed_count'] / df_delayed_route['total_flights'] * 100
    )

    # Sort the routes by the delayed percentage (ascending order)
    df_delayed_route = df_delayed_route.sort_values("percentage", ascending=True)

    # --- Merge with route cities mapping ---

    # Reset index so that 'route_code' becomes a column
    df_delayed_route = df_delayed_route.reset_index()

    # Merge with df_route_cities to get the user-friendly city names
    # (Assuming df_route_cities has columns 'route_icao' and 'route_cities')
    df_delayed_route = df_delayed_route.merge(
        df_route_cities,
    )

    # --- Create the Plotly bar chart using route_cities as the bar names ---

    fig = px.bar(
        df_delayed_route,
        x="percentage",
        y="route_cities",  # Use the user-friendly route cities for the y-axis
        orientation="h",
        title="Flights Delayed at Origin by Route",
        labels={"percentage": "Percentage", "route_cities": "Route"},
        text=df_delayed_route["delayed_count"].astype(str) + " out of " + df_delayed_route["total_flights"].astype(str),
        color="percentage",  # Color by percentage of delays
        color_continuous_scale='reds'
    )

    # Calculate the average delayed percentage and add a vertical average line
    avg_route_percentage = df_delayed_route["percentage"].mean()
    fig.add_vline(
        x=avg_route_percentage,
        line=dict(color="black", dash="dash"),
        annotation_text=f"Average: {avg_route_percentage:.1f}%",
        annotation_position="bottom right"
    )

    # Update layout for better presentation
    fig.update_layout(
        xaxis_title="Percentage of Delayed Flights",
        yaxis_title="Route",
        showlegend=True,
        width=1000,
        height=1200,
        xaxis_title_font_size=20,
        yaxis_title_font_size=20
    )

    fig.update_traces(
        textfont=dict(size=30),
        textposition="outside",
        cliponaxis=False
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


# ---------- Delay Map Page ----------

elif selected == "Delay Map":
    PAGE_TITLE = 'Delays Per Airport'
    PAGE_SUB_TITLE = 'Departure delays of selected aiports and routes, ranging from 2022 to 2024'

    st.title(PAGE_TITLE)
    st.caption(PAGE_SUB_TITLE)
    def display_airport_facts(df, icao, metric_title, graph_selection={}):
        if metric_title == 'Airport Name':
            name = df[df['origin.code_icao'] == icao].iloc[0]['origin_airport_name']
            metric = f"{name} ({icao}) in {df_route_cities[df_route_cities['origin.icao_code'] == icao].iloc[0]['origin']}"
            st.metric(metric_title,metric)
        elif 'Percentage' in metric_title:
            percentage_one_decimal = round(df[df["origin.code_icao"] == icao][metric_title].iloc[0],1)
            metric = f'{percentage_one_decimal}%'
            st.metric(metric_title,metric)
        elif metric_title in ['Routes']: #,'Operators','Aircrafts']:

            # Define the correct column name based on metric_title
            column_mapping = {
                "Routes": "ICAO_route",
                "Operators": "operator_icao",
                "Aircrafts": "aircraft_type"
            }
            selected_column = column_mapping.get(metric_title, None)

            if selected_column:
                # Get all unique values for the selected category
                available_options = df[df["origin.code_icao"] == icao][selected_column].unique()
                # Define a unique key for the multiselect widget
                multiselect_key = f"multiselect_{selected_column}"

                # Initialize the session state for the multiselect if not already set
                if multiselect_key not in st.session_state:
                    st.session_state[multiselect_key] = list(available_options)

                # Add a "Select All" button
                if st.button(f"Select All {metric_title}"):
                    st.session_state[multiselect_key] = list(available_options)
                # Store selections
                current_selection = st.multiselect(
                    f"Choose {metric_title}:",
                    options=available_options,
                    key=multiselect_key
                )
                return {selected_column:current_selection}

        elif metric_title == 'Graph':
            # Ensure selections exist
            selected_routes = st.session_state.get("Routes", [])
            # selected_operators = st.session_state.get("Operator", [])
            # selected_aircraft = st.session_state.get("Aircraft", [])
            fig = rene.departure_delay_prog_for_route_group(df,icao,regionalize=False,routes=graph_selection["ICAO_route"]) #, operators=graph_selection["operator_icao"], aircraft_types=graph_selection["aircraft_type"])
            st.plotly_chart(fig, use_container_width=True)
        else:
            metric = df[df["origin.code_icao"] == icao][metric_title]
            st.metric(metric_title,metric)

    def display_map(df,df_airport_info):
            # Exponential scaling function for normalization
        def exponential_normalization(value, min_val, max_val, base=3):
            if min_val == max_val:
                return 0  # Prevent division by zero
            return (np.log1p(value - min_val) / np.log1p(max_val - min_val)) / np.log(base)

        # Set up color map with exponential normalization
        min_delay, max_delay = df['Percentage of Departure Delays'].min(), df['Percentage of Departure Delays'].max()

        colormap = colormaps['RdYlGn']

        # Function to get exponential normalized color
        def get_exponential_color(value):
            normalized_value = exponential_normalization(value, min_delay, max_delay)
            color = colormap(1 - normalized_value)  # Invert scale for color mapping
            return mcolors.rgb2hex(color)

        # Function to enhance color saturation
        def enhance_saturation(hex_color, factor=1.5):
            rgb = mcolors.hex2color(hex_color)
            h, l, s = colorsys.rgb_to_hls(*rgb)
            s = min(1, s * factor)  # Ensure saturation doesn't exceed 1
            enhanced_rgb = colorsys.hls_to_rgb(h, l, s)
            return mcolors.rgb2hex(enhanced_rgb)

        # Create a map centered around a central location
        m = folium.Map(location=[48.3328, -8.7853], zoom_start=2, tiles='CartoDB positron')

        # Add circle markers for each airport
        for idx, row in df.iterrows():
            color = get_exponential_color(row['Percentage of Departure Delays'])
            saturated_color = enhance_saturation(color, factor=1.5)
            size = row['Number of Flights'] / 500  # Adjust dot size
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; font-size: 14px;">
                <strong>{row['origin.code_icao']}</strong><br>
                <span style="color: #d9534f;">{row['Percentage of Departure Delays']:.1f}% delayed</span><br>
                <span>{row['Number of Flights']} flights</span>
            </div>
            """
            popup = folium.Popup(popup_html, max_width=250)
            folium.CircleMarker(
                location=(row['origin_airport_lat'], row['origin_airport_lon']),
                radius=size,
                color=saturated_color,
                weight=1,
                fill=True,
                fill_color=saturated_color,
                fill_opacity=0.6,
                tooltip=df_airport_info[df_airport_info['origin.code_icao'] == row["origin.code_icao"]].iloc[0]['origin_airport_name'],
                popup=f"{row['origin.code_icao']}\n{row['Percentage of Departure Delays']:.1f}% delayed\n{row['Number of Flights']} flights"
            ).add_to(m)

        col_map1, col_map2 = st.columns([1, 0.000001])
        
        # Display the map
        with col_map1:
            st.markdown('## Map of Flight Delays')
            st_map = st_folium(m,width=1200,height=500)

        # st.write(st_map)
        # st.write('latitude:',st_map["last_object_clicked"]["lat"])
        # st.write('longitude:',st_map["last_object_clicked"]["lng"])
        try:
            clicked_lat = st_map["last_object_clicked"]["lat"]
            clicked_lng = st_map["last_object_clicked"]["lng"]

            # Compute the Euclidean distance between the clicked point and each airport's coordinates.
            # (For short distances, this is a reasonable approximation.)
            distances = np.sqrt(
                (df['origin_airport_lat'] - clicked_lat) ** 2 +
                (df['origin_airport_lon'] - clicked_lng) ** 2
            )

            # Identify the row (airport) with the minimum distance.
            closest_idx = distances.idxmin()
            current_icao = df.loc[closest_idx, 'origin.code_icao']

            # DISPLAY FACTS
            st.markdown('### Airport Facts')

            with st.container(border=1):
                display_airport_facts(df_airport_info, current_icao, "Airport Name")

            first_row_metrics = [
                ('Percentage of Departure Delays', df),
                ('Delayed Flights', df),
                ('Number of Flights', df)
            ]

            # Create a row with as many columns as metrics
            cols_first_row = st.columns(len(first_row_metrics))
            for col, (metric_title, data_source) in zip(cols_first_row, first_row_metrics):
                with col.container(border=1):
                    display_airport_facts(data_source, current_icao, metric_title)

            # -------------------------------
            # Second row: Display the Graph and the side metrics
            # -------------------------------
            # Create two columns: left for the last three metrics and right for the Graph.
            # You can adjust the relative widths as needed.
            left_col, right_col = st.columns([1, 2])
            current_selection = {}
            # Left column: Display the last three metrics vertically
            side_metrics = [
                ('Routes', df_airport_info),]
            #     ('Operators', df_airport_info),
            #     ('Aircrafts', df_airport_info)
            # ]

            with left_col:
                st.markdown("### Additional Info")
                for metric_title, data_source in side_metrics:
                    # You can separate each metric with a divider or some spacing
                    with st.container(border=1):
                        result = display_airport_facts(data_source, current_icao, metric_title)
                        # If display_airport_facts returns a dict that you want to merge,
                        # you can update current_selection:
                        if isinstance(result, dict):
                            current_selection.update(result)
            
            with right_col:
                st.markdown("### Graph")
                with st.container(border=1):
                    try:
                        display_airport_facts(df_airport_info, current_icao, 'Graph',graph_selection=current_selection)
                    except:
                        st.write('No graph available for current combination of routes, operators, and aircrafts.')
            # ---------- Secret Button for LFPO (Orly) ----------
            if current_icao == "LFPO":  # Show only if LFPO is selected
                st.markdown(
                    """
                    <style>
                    .hidden-button button {
                        font-size: 3px !important;  /* Make text super small */
                        padding: 2px 5px !important;  /* Reduce padding */
                        background-color: rgba(0,0,0,0) !important;  /* Transparent background */
                        color: rgba(0,0,0,0) !important;  /* Invisible text */
                        border: none !important;  /* Remove border */
                        cursor: pointer;
                    }
                    .hidden-button button:hover {
                        background-color: rgba(255, 255, 255, 0.1) !important;  /* Slight hover effect */
                        color: #333 !important;  /* Show text slightly on hover */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                # Add a super small button with hidden appearance
                st.markdown('<div class="hidden-button">', unsafe_allow_html=True)
                if st.button(type="tertiary", label=".", key="hidden_button"):
                    st.audio("streamlit/streamlit_songs/Orly.mp3", format="audio/mp3")
                st.markdown("</div>", unsafe_allow_html=True)

            # # Create an expander (dropdown-style)
            # with st.expander("Select Routes", expanded=False):
            #     option1 = st.selectbox("Select an Airport", ["JFK", "LAX", "ORD", "SFO"], index=0)
            #     option2 = st.selectbox("Select a Flight Status", ["On Time", "Delayed", "Cancelled"], index=0)
            #     option3 = st.selectbox("Select a Weather Condition", ["Clear", "Rain", "Storm", "Fog"], index=0)

            # # Display selected values
            # st.write(f"**Selected Airport:** {option1}")
            # st.write(f"**Selected Flight Status:** {option2}")
            # st.write(f"**Selected Weather Condition:** {option3}")
        except Exception as e:
            st.write('Select an airport on the map for more details...')

    # DISPLAY MAP
    display_map(df,df_names_routes)

# ---------- Additional Pages (Delay Prediction, Contact) ----------
elif selected == "Prediction":
    st.title("Delay Prediction")
    st.caption("Predicting flight delays, given a weather report.")
    # st.caption("NOTE: All of the probability values on this page are conditional probabilities, based on the input features of our model. It is not a prediction of the total delay probability.")
   
    def display_airport_facts(df, icao, metric_title, graph_selection={}, threshold=0.5):
        if metric_title == 'Airport Name':
            name = df[df['origin.code_icao'] == icao].iloc[0]['origin_airport_name']
            metric = f"{name} ({icao}) in {df_route_cities[df_route_cities['origin.icao_code'] == icao].iloc[0]['origin']}"
            st.metric(metric_title,metric)
        elif metric_title == 'Routes':
            # Define the correct column name based on metric_title
            column_mapping = {
                "Routes": "ICAO_route"
            }
            selected_column = column_mapping.get(metric_title, None)

            if selected_column:
                # Get all unique values for the selected category
                available_options = df[df["origin.code_icao"] == icao][selected_column].unique()

                # Define a unique key for the selectbox widget
                selectbox_key = f"selectbox_{selected_column}"

                # Initialize the session state for the selectbox if not already set
                if selectbox_key not in st.session_state:
                    st.session_state[selectbox_key] = available_options[0] if len(available_options) > 0 else None  # Default to first option if available

                # Store selection (single choice dropdown)
                current_selection = st.selectbox(
                    f"Choose Route:",
                    options=available_options,
                    key=selectbox_key
                )

                st.markdown("---")

                stats,_,_,_ = rene2.streamlit_prediction_stats(df_metar,current_selection,threshold)
                
                stats.rename(columns={
                    "route": "Route",
                    "total_flights": "Total Flights",
                    "total_delayed": "Delayed Flights",
                    "total_ontime": "On Time Flights",
                    "percent_delayed": "Percentage Delayed"
                }, inplace=True)

                stats['Percentage Delayed'] = stats['Percentage Delayed'].apply(lambda x: f"{x:.1%}")

                first_row = stats.iloc[0]

                for col in stats.columns[:5]:
                    st.markdown(f"**{col}**: {first_row[col]}")

                return {selected_column: current_selection}
        
        elif metric_title == 'METAR':
            # Get METAR and corresponding ML_index
            current_metars_indeces_list = df[df['route_code'] == graph_selection['ICAO_route']][['METAR_departure_ref', 'ML_index']]

            # Convert to dictionary for lookup (METAR -> ML_index)
            metar_to_index_dict = dict(zip(current_metars_indeces_list['METAR_departure_ref'], current_metars_indeces_list['ML_index']))

            # Define a unique key for the selectbox widget
            multiselect_key = "multiselect_metar"

            # Initialize the session state for the selectbox if not already set
            if multiselect_key not in st.session_state:
                st.session_state[multiselect_key] = None  # No default selection

            # Store selections
            current_selection = st.selectbox(
                label="Choose Weather Report Departure Reference:",
                label_visibility="collapsed",
                options=list(metar_to_index_dict.keys()),  # Use METAR references as dropdown options
                key=multiselect_key,
            )

            # Get the corresponding ML_index
            current_index = metar_to_index_dict.get(current_selection, None)

            return {'METAR':current_selection,'ML_index':current_index}            

        elif metric_title == 'Histogram':
            st.markdown('#### Predicted Probabilities of Departure Delay')
            st.caption(f"Route: {graph_selection['ICAO_route']}")
            # Add a threshold slider to Streamlit
            threshold = st.slider("Set Probability Threshold:", 0.5, 1.0, 0.5, 0.1)
            _,fig,_,_ = rene2.streamlit_prediction_stats(df,graph_selection['ICAO_route'], threshold)
            # st.write(type(fig))
            st.plotly_chart(fig, use_container_width=True)
            return threshold

        elif metric_title == 'Pie Charts':
            st.markdown('#### Prediction Quality')
            st.caption(f"Route: {graph_selection['ICAO_route']}")
            _,_,pie1,pie2 = rene2.streamlit_prediction_stats(df,graph_selection['ICAO_route'], threshold)
            # st.write(type(fig))
            st.plotly_chart(pie1, use_container_width=True)
            st.plotly_chart(pie2, use_container_width=True)

        elif metric_title == 'Weather Report':
            if graph_selection['METAR'] is None:
                st.write("Select a METAR for more details...")
            else:
                try:
                    decoded_data = smp.parse_metar(graph_selection['METAR'])
                    if "Error" in decoded_data:
                        st.error("Weather Report not available.")
                    else:
                        for key, value in decoded_data.items():
                            st.write(f"**{key}:** {value}")
                except:
                    st.write("Weather Report not available. Please try a different METAR.")

        elif metric_title == 'Delay Prediction':
            try:
                probability = round((df[df['ML_index'] == graph_selection['ML_index']]['predicted_prob_class_1'].iloc[0])*100,1)
                st.write(f"Conditional Probability of Delay on Departure:")
                st.markdown(f"## {probability}%")
            except:
                st.write('Select a METAR for more details...')

            # airport_name = df[df['origin.code_icao'] == icao].iloc[0]['origin_airport_name']


        else:
            # metric = df[df["origin.code_icao"] == icao][metric_title]
            # st.metric(metric_title,metric)
            st.write(metric_title)


    def display_map(df,df_airport_info):
        # Exponential scaling function for normalization
        def exponential_normalization(value, min_val, max_val, base=3):
            if min_val == max_val:
                return 0  # Prevent division by zero
            return (np.log1p(value - min_val) / np.log1p(max_val - min_val)) / np.log(base)

        # Set up color map with exponential normalization
        min_delay, max_delay = df['Percentage of Departure Delays'].min(), df['Percentage of Departure Delays'].max()

        colormap = colormaps['RdYlGn']

        # Function to get exponential normalized color
        def get_exponential_color(value):
            normalized_value = exponential_normalization(value, min_delay, max_delay)
            color = colormap(1 - normalized_value)  # Invert scale for color mapping
            return mcolors.rgb2hex(color)

        # Function to enhance color saturation
        def enhance_saturation(hex_color, factor=1.5):
            rgb = mcolors.hex2color(hex_color)
            h, l, s = colorsys.rgb_to_hls(*rgb)
            s = min(1, s * factor)  # Ensure saturation doesn't exceed 1
            enhanced_rgb = colorsys.hls_to_rgb(h, l, s)
            return mcolors.rgb2hex(enhanced_rgb)

        # Create a map centered around a central location
        m = folium.Map(location=[48.3328, -8.7853], zoom_start=2, tiles='CartoDB positron')
        
        # Add circle markers for each airport
        for idx, row in df.iterrows():
            color = get_exponential_color(row['Percentage of Departure Delays'])
            saturated_color = enhance_saturation(color, factor=1.5)
            size = row['Number of Flights'] / 500  # Adjust dot size
            popup_html = f"""
            <div style="font-family: Arial, sans-serif; font-size: 14px;">
                <strong>{row['origin.code_icao']}</strong><br>
                <span style="color: #d9534f;">{row['Percentage of Departure Delays']:.1f}% delayed</span><br>
                <span>{row['Number of Flights']} flights</span>
            </div>
            """
            popup = folium.Popup(popup_html, max_width=250)
            folium.CircleMarker(
                location=(row['origin_airport_lat'], row['origin_airport_lon']),
                radius=size,
                color=saturated_color,
                weight=1,
                fill=True,
                fill_color=saturated_color,
                fill_opacity=0.6,
                tooltip=df_airport_info[df_airport_info['origin.code_icao'] == row["origin.code_icao"]].iloc[0]['origin_airport_name'],
                popup=popup
            ).add_to(m)

        col_map1, col_map2 = st.columns([1, 0.000001])
        
        # Display the map
        with col_map1:
            st.markdown('## Map of Flight Delays')
            st_map = st_folium(m,width=1200,height=500)
        
        try:
            clicked_lat = st_map["last_object_clicked"]["lat"]
            clicked_lng = st_map["last_object_clicked"]["lng"]

            # Compute the Euclidean distance between the clicked point and each airport's coordinates.
            # (For short distances, this is a reasonable approximation.)
            distances = np.sqrt(
                (df['origin_airport_lat'] - clicked_lat) ** 2 +
                (df['origin_airport_lon'] - clicked_lng) ** 2
            )

            # Identify the row (airport) with the minimum distance.
            closest_idx = distances.idxmin()
            current_icao = df.loc[closest_idx, 'origin.code_icao']

            with st.container(border=1):
                display_airport_facts(df_airport_info, current_icao, "Airport Name")

            # First Row: Routes, Histogram, Pie Charts
            cols_first_row = st.columns([1,4,2])

            with cols_first_row[0]:
                with st.container(border=1):
                    st.markdown('#### Choose a Route')
                    result = display_airport_facts(df_airport_info, current_icao, 'Routes')
                    # Store selection if it's a dictionary
                    current_selection = result if isinstance(result, dict) else {}

            with cols_first_row[1]:
                with st.container(border=1):
                    try:
                        threshold = display_airport_facts(df_metar, current_icao, 'Histogram', graph_selection=current_selection)
                    except Exception as e:
                        st.write('Select a route for more details...')

            with cols_first_row[2]:
                with st.container(border=1):
                    try:
                        display_airport_facts(df_metar, current_icao, 'Pie Charts', current_selection, threshold)
                    except Exception as e:
                        st.write('Select a route for more details...')
            
            st.markdown("---")

            # Second Row: METAR Input and Delay Prediction
            cols_second_row = st.columns([3,3,2])

            with cols_second_row[0]:  # METAR Input (1/2)
                with st.container(border=1):
                    st.markdown('#### Choose a Weather Report (METAR code)')
                    result = display_airport_facts(df_metar, current_icao, 'METAR', current_selection)
                    if isinstance(result, dict):
                        current_selection.update(result)

            with cols_second_row[1]:  # Delay Prediction (1/2)
                with st.container(border=1):
                    st.markdown('#### Weather Report')
                    display_airport_facts(df_metar, current_icao, 'Weather Report', current_selection)


            with cols_second_row[2]:  # Weather Report (1/2)
                with st.container(border=1):
                    st.markdown('#### Delay Prediction')
                    display_airport_facts(df_metar, current_icao, 'Delay Prediction', current_selection)
        except Exception as e:
            # st.write(e)
            st.write('Select an airport on the map for more details...')              
    # DISPLAY MAP
    display_map(df,df_names_routes)    
    
elif selected == "Contact":
    st.title("Meet the team!")
    st.write("Connect with us on LinkedIn")
    
    linkedin_logo_url = "https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png"
    
    cols = st.columns(3)

    with cols[0]:
        st.header("René Marcel Falquier")
        st.image(linkedin_logo_url, width=50)
        st.image(
            "streamlit/streamlit_img/linkedin_rene.jpg",
            use_container_width=True
        )
        st.caption("LinkedIn QR Code")

    with cols[1]:
        st.header("Martina Wengle")
        st.image(linkedin_logo_url, width=50)
        st.image(
            "streamlit/streamlit_img/linkedin_martina.jpg",
            use_container_width=True
        )
        st.caption("LinkedIn QR Code")

    with cols[2]:
        st.header("Ralf Reuvers")
        st.image(linkedin_logo_url, width=50)
        st.image(
            "streamlit/streamlit_img/linkedin_ralf.jpg",
            use_container_width=True
        )
        st.caption("LinkedIn QR Code")
    
    st.markdown("---")

    # ---------- Technical Advisor & Project Sponsor ----------
    st.markdown("## **Advisors & Sponsors**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### **Technical Advisor**")
        st.write("Kristjan Rognvaldsson")
        st.write("*CVE Structures & Interiors*")

    with col2:
        st.markdown("### **Project Sponsor**")
        st.write("Mike Vergalla")
        st.write("*Free Flight Lab*")

    # st.markdown("---")
    
    # # ---------- Songs Section ----------
    # with st.expander("🎵 **Songs - Click to Expand** 🎶", expanded=False):
    #     st.write("Here are some songs to listen to while you explore our app:")

    #     st.write("### Dependency Issues")
    #     st.audio("data/songs/Dependency Issues.mp3", format="audio/mp3")

    #     st.write("### Noverbosity")
    #     st.audio("data/songs/noverbosity.mp3", format="audio/mp3")

    #     st.write("### Outta RAM")
    #     st.audio("data/songs/Outta RAM.mp3", format="audio/mp3")

    #     st.write("### Rene")
    #     st.audio("data/songs/Rene.mp3", format="audio/mp3")

    #     st.write("### SHAP et XGBoost")
    #     st.audio("data/songs/SHAP et XGBoost.mp3", format="audio/mp3")
