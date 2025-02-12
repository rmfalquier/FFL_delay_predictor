import pandas as pd
import ast
from tqdm import tqdm

def metar_cleaning(metar_data_df):
    """Cleans the METAR data DataFrame by performing the following steps:
    1. Drop duplicates and reset index
    2. Replace empty lists in 'wind_variable_direction' with 0, otherwise 1
    3. One-hot encode 'wx_codes' column
    4. One-hot encode 'pressure_tendency' column
    5. Normalize 'altimeter' column to hPa
    6. Remove rows where 'temperature' is greater than 80
    7. Normalize 'visibility' column to meters
    8. Drop unnecessary columns
    9. One-hot encode 'categories' column
    10. Normalize 'altimeter' column to hPa
    11. Parse and map weather codes to categories
    12. One-hot encode the categories
    13. One-hot encode pressure_tendency
    14. Re-order columns
    15. Normalize 'visibility' column
    16. Parse and categorize 'clouds' with numerical encoding
    17. Remove rows where 'temperature' is NaN
    18. Apply mapping to 'flight_rules' column
    19. Remove columns that start with 'remarks_info.precip'
    20. Remove unnecessary columns
    21. Impute missing values
    22. Build a visibility map based on cloud layers
    23. Impute visibility based on the map
    24. Build a visibility map based on flight rules
    25. Impute visibility based on flight rules
    26. Impute remaining missing values using flight rules
    27. Reset index
    28. Drop rows with missing values in specific columns

    Args:
        metar_data_df (DataFrame): DataFrame containing METAR data

    Returns:
        DataFrame: Cleaned METAR data DataFrame
    """

    # Display progress bar
    tqdm.pandas(desc="Cleaning METAR data...")

    # Drop duplicates and reset index
    metar_data_df.drop_duplicates(inplace=True)
    metar_data_df.reset_index(inplace=True,drop=True)
    if 'Unnamed: 0' in metar_data_df.columns:
        metar_data_df.drop(columns='Unnamed: 0',inplace=True)

    # Replace empty lists in wind_variable_direction with 0, otherwise 1
    metar_data_df['wind_variable_change'] = metar_data_df['wind_variable_direction'].progress_apply(lambda x: 0 if len(x) == 2 else 1)
    metar_data_df.drop(columns='wind_variable_direction', inplace=True)

    # One-hot encode wx_codes
    # Mapping conditions to categories (including multi-category mappings)
    condition_categories = {
        'blowing snow': ['Blowing Snow', 'Low Drifting Snow'],
        'blowing dust': ['Blowing Wide Dust', 'Sand', 'Wide Dust'],
        'light rain': ['Drizzle', 'Drizzle Rain', 'Light Drizzle', 'Light Drizzle Rain', 'Rain Drizzle', 'Light Rain',
                    'Light Rain Drizzle', 'Light Showers', 'Light Showers Rain'],
        'heavy rain': ['Heavy Rain', 'Heavy Showers Rain', 'Showers Rain'],
        'fog': ['Fog', 'Patchy Fog', 'Shallow Fog'],
        'light fog': ['Mist', 'Partial Fog'],
        'light hail': ['Light Ice Pellets', 'Light Showers Small Hail'],
        'hail': ['Showers Small Hail', 'Heavy Thunderstorm Rain Hail', 'Heavy Thunderstorm Hail Rain'],
        'light snow': ['Light Snow', 'Light Drizzle Snow', 'Light Drizzle Snow Grains',
                    'Light Snow Grains', 'Light Snow Grains Drizzle', 'Light Showers Snow'],
        'heavy snow': ['Heavy Snow', 'Showers Snow'],
        'rain': ['Rain', 'Thunderstorm Rain', 'Light Ice Pellets Rain'],
        'snow': ['Snow', 'Snow Rain'],
        'thunderstorm': ['Thunderstorm', 'Heavy Thunderstorm Rain', 'Heavy Thunderstorm Rain Hail',
                        'Heavy Thunderstorm Hail Rain', 'Thunderstorm Vicinity Showers', 'Vicinity Thunderstorm',
                        'Light Thunderstorm Rain'],
        'vicinity showers': ['Vicinity Showers'],
        'vicinity fog': ['Vicinity Fog'],
        'funnel cloud': ['Funnel Cloud'],
        'haze': ['Haze'],
        'smoke': ['Smoke'],
        'freezing': ['Freezing Fog', 'Freezing Drizzle', 'Light Freezing Drizzle','Light Freezing Drizzle Snow']
    }

    # Reverse the mapping to simplify lookup
    condition_to_category = {}
    for category, conditions in condition_categories.items():
        for condition in conditions:
            if condition not in condition_to_category:
                condition_to_category[condition] = []
            condition_to_category[condition].append(category)

    # Function to parse and map weather codes to categories
    def map_conditions_to_categories(wx_codes):
        try:
            # Parse the string into a Python list
            parsed = ast.literal_eval(wx_codes)
            # Extract and map to categories
            categories = set()
            for item in parsed:
                if isinstance(item, dict) and 'value' in item:
                    condition = item['value']
                    if condition in condition_to_category:
                        categories.update(condition_to_category[condition])
            return list(categories)
        except (ValueError, SyntaxError):
            return []
        
    # Apply parsing and mapping to the 'wx_codes' column
    metar_data_df['categories'] = metar_data_df['wx_codes'].progress_apply(map_conditions_to_categories)

    # One-hot encode the categories
    category_columns = pd.get_dummies(metar_data_df['categories'].progress_apply(pd.Series).stack()).groupby(level=0).sum()

    # Ensure integer type for one-hot encoded columns
    category_columns = category_columns.astype(int)

    # Add prefix 'wx_code_' to all columns
    category_columns.columns = [f"wx_code_{str.replace(col,' ','_')}" for col in category_columns.columns]

    # Fill missing rows with zeros (ensures no missing data for rows without wx codes)
    category_columns = category_columns.reindex(metar_data_df.index, fill_value=0)

    # Add the one-hot encoded columns to the DataFrame
    metar_data_df = pd.concat([metar_data_df, category_columns], axis=1)

    # Drop the original 'wx_codes' column
    metar_data_df.drop(columns=['wx_codes','categories'], inplace=True)

    # One-hot encoding pressure_tendency
    tendency_dummies = pd.get_dummies(
        metar_data_df['remarks_info.pressure_tendency.tendency'], 
        prefix='pressure_tendency'
    )

    # Clean column names
    tendency_dummies.columns = (
        tendency_dummies.columns.str.replace(' ', '_', regex=False)
                                .str.replace(',', '', regex=False)
                                .str.lower()
    )

    # Ensure one-hot encoded columns are integers (0/1)
    tendency_dummies = tendency_dummies.astype(int)

    # Add cleaned column names back
    metar_data_df = pd.concat([metar_data_df, tendency_dummies], axis=1)

    # Drop the original column
    metar_data_df.drop(columns=['remarks_info.pressure_tendency.tendency'], inplace=True)

    metar_data_df.dropna(axis=1,how='all', inplace=True)

    metar_data_df.columns = metar_data_df.columns.str.replace('.value', '', regex=False)

    # Remove rows where 'temperature' is greater than 80
    metar_data_df = metar_data_df[metar_data_df['temperature'] <= 80]  

    def normalize_altimeter(value):
        # Treat values < 60 as inches of mercury (e.g., 30 for 30.00 inHg)
        if 0 <= value < 100:
            return value * 33.8639  # Convert inHg to hPa
        elif value >= 900:
            return value  # Already in hPa
        else:
            return None  # Invalid value

    # Apply normalization
    metar_data_df['altimeter_hpa'] = metar_data_df['altimeter'].progress_apply(normalize_altimeter)
    metar_data_df.drop(columns=['altimeter'], inplace=True)

    # Sanitized is a reduced version of raw, so drop it.
    metar_data_df.drop(columns=['sanitized'], inplace=True)

    # Re-ordering columns

    # List of prioritized columns
    first_columns = ['raw', 'time.dt', 'station']

    # Separate columns without missing values
    non_missing_columns = [
        col for col in metar_data_df.columns if col not in first_columns and not metar_data_df[col].isnull().any()
    ]

    # Separate columns with missing values
    remaining_columns = [
        col for col in metar_data_df.columns if col not in first_columns and col not in non_missing_columns
    ]

    # Combine all columns: prioritized + non-missing + remaining
    reordered_columns = first_columns + non_missing_columns + remaining_columns

    # Reorder the DataFrame
    metar_data_df = metar_data_df[reordered_columns]

    def normalize_visibility(value):
        if value > 10:  # Assume values greater than 10 are in meters
            return value  # Keep as meters
        else:  # Assume values <= 10 are in statute miles
            return value * 1609.34  # Convert miles to meters

    # Apply normalization to the visibility column
    metar_data_df['visibility_meters'] = metar_data_df['visibility'].progress_apply(normalize_visibility).clip(lower=50, upper=9999)

    metar_data_df.drop(columns=['visibility'], inplace=True)  # Drop the original column

    metar_data_df.drop(columns=['wind_direction', 'remarks', 'remarks_info.codes'], inplace=True)

    # Replace missing values in the 'wind_gust' column with 0
    metar_data_df['wind_gust'] = metar_data_df['wind_gust'].fillna(0)

    # Replace missing values in the 'wind_speed' column with 0
    metar_data_df['wind_speed'] = metar_data_df['wind_speed'].fillna(0)

    # Group by 'station' and fill missing values with the mean for each station
    metar_data_df['remarks_info.sea_level_pressure'] = metar_data_df.groupby('station')[
        'remarks_info.sea_level_pressure'].transform(lambda x: x.fillna(x.mean()))
    
    # Impute missing values with the median of the column
    metar_data_df['remarks_info.sea_level_pressure'] = metar_data_df['remarks_info.sea_level_pressure'].fillna(
        metar_data_df['remarks_info.sea_level_pressure'].median()
    )

    # Define mappings for numerical encoding
    altitude_category_mapping = {'low': 1, 'medium': 2, 'high': 3, 'vertical': 4, 'unknown': 0}
    cloud_type_mapping = {'CLR': 1, 'FEW': 2, 'SCT': 3, 'BKN': 4, 'OVC': 5, 'unknown': 0}

    # # Replace missing values in the 'clouds' column with an empty list
    # metar_data_df['clouds'] = metar_data_df['clouds'].apply(lambda x: [] if pd.isna(x) else x)

    # Function to parse the stringified clouds column
    def parse_clouds_column(value):
        return ast.literal_eval(value)  # Safely parse the string into a Python list
    
    metar_data_df['clouds'] = metar_data_df['clouds'].progress_apply(parse_clouds_column)

    # Function to parse and categorize clouds with numerical encoding
    def parse_and_categorize_clouds_numeric(clouds):
        if not isinstance(clouds, list):
            return {}

        # Parse each cloud layer
        parsed = {}
        for i, cloud in enumerate(clouds):
            layer = f"clouds_layer_{i + 1}"  # Positional layer (layer_1, layer_2, ...)
            altitude = cloud.get('altitude', None)
            if altitude is not None:
                if altitude <= 65:
                    altitude_category = 'low'
                elif altitude <= 200:
                    altitude_category = 'medium'
                elif altitude > 200:
                    altitude_category = 'high'
                else:
                    altitude_category = 'unknown'
            else:
                altitude_category = 'unknown'

            # Check for vertical clouds
            if cloud.get('modifier') in ['CB', 'TCU']:
                altitude_category = 'vertical'

            # Encode using numerical mapping
            parsed[f"{layer}_type"] = cloud_type_mapping.get(cloud.get('type', 'unknown'), 0)
            parsed[f"{layer}_altitude_category"] = altitude_category_mapping.get(altitude_category, 0)

        return parsed

    # Apply parsing to the 'clouds' column
    parsed_clouds = metar_data_df['clouds'].progress_apply(parse_and_categorize_clouds_numeric)

    # Convert parsed cloud data into a DataFrame
    parsed_clouds_df = pd.DataFrame(parsed_clouds.tolist())

    # Replace NaN with 0
    parsed_clouds_df = parsed_clouds_df.fillna(0)

    # Combine the parsed data back with the original DataFrame
    metar_data_df.reset_index(inplace=True, drop=True)
    metar_data_df = pd.concat([metar_data_df, parsed_clouds_df], axis=1)

    # Remove Clouds column
    metar_data_df.drop(columns=['clouds'], inplace=True)

    # Drop rows where 'temperature' is NaN
    metar_data_df = metar_data_df.dropna(subset=['temperature'])

    # Define severity mapping for flight rules
    flight_rules_mapping = {'VFR': 1, 'MVFR': 2, 'IFR': 3, 'LIFR': 4}

    # Apply the mapping to the 'flight_rules' column
    metar_data_df['flight_rules'] = metar_data_df['flight_rules'].map(flight_rules_mapping)

    # Remove columns that start with remarks_info.precip
    metar_data_df = metar_data_df.loc[:, ~metar_data_df.columns.str.startswith('remarks_info.precip')]

    # Remove unnecessary columns
    metar_data_df = metar_data_df.drop(columns=[
        'remarks_info.temperature_decimal', # Already have temperature column
        'remarks_info.pressure_tendency.change', # Already have one-hot encoded pressure tendency columns
        'remarks_info.maximum_temperature_6', # Already have temperature column
        'remarks_info.dewpoint_decimal', # Already have dewpoint column
        'remarks_info.minimum_temperature_6', # Already have temperature column
        'remarks_info.minimum_temperature_24', # Already have temperature column
        'remarks_info.maximum_temperature_24', # Already have temperature column
        'remarks_info.snow_depth', # wx_codes should cover snow
        'runway_visibility', # Already have visibility column
        'visibility.normalized', # Already have visibility column
        'visibility.numerator', # Already have visibility column
        'visibility.denominator', # Already have visibility column
        'other' # Mixed information about clouds, weather, etc. We already have enough information about them.
    ])

    # Step 1: Create a visibility map based on layer_1_type and layer_1_altitude_category
    def build_visibility_map(df):
        # Group by layer_1_type and layer_1_altitude_category and calculate median visibility
        visibility_map = (
            df.groupby(['clouds_layer_1_type', 'clouds_layer_1_altitude_category'])['visibility_meters']
            .median()
            .dropna()
            .to_dict()
        )
        return visibility_map

    # Step 2: Function to impute visibility based on the map
    def impute_visibility(row, visibility_map):
        # If clouds_layer_1_type or clouds_layer_1_altitude_category is 0, leave as NaN
        if row['clouds_layer_1_type'] == 0 and row['clouds_layer_1_altitude_category'] == 0:
            return row['visibility_meters']

        # Otherwise, use the visibility map to impute
        key = (row['clouds_layer_1_type'], row['clouds_layer_1_altitude_category'])
        if pd.isna(row['visibility_meters']) and key in visibility_map:
            return visibility_map[key]

        # If no match is found in the map, leave as NaN
        return row['visibility_meters']

    # Step 3: Build the visibility map
    visibility_map = build_visibility_map(metar_data_df)

    # Step 4: Apply the imputation function to the DataFrame
    metar_data_df['visibility_meters'] = metar_data_df.progress_apply(
        lambda row: impute_visibility(row, visibility_map), axis=1
    )

    # Step 1: Create a visibility map based on flight_rules
    def build_flight_rules_visibility_map(df):
        # Group by flight_rules and calculate median visibility
        visibility_map = (
            df.groupby('flight_rules')['visibility_meters']
            .median()
            .dropna()
            .to_dict()
        )
        return visibility_map

    # Step 2: Function to impute remaining missing values using flight_rules
    def impute_visibility_with_flight_rules(row, flight_rules_map):
        # If visibility is still NaN, impute based on flight_rules
        if pd.isna(row['visibility_meters']) and row['flight_rules'] in flight_rules_map:
            return flight_rules_map[row['flight_rules']]

        # Otherwise, return the original value
        return row['visibility_meters']

    # Step 3: Build the flight_rules visibility map
    flight_rules_map = build_flight_rules_visibility_map(metar_data_df)

    # Step 4: Impute remaining missing values
    metar_data_df['visibility_meters'] = metar_data_df.progress_apply(
        lambda row: impute_visibility_with_flight_rules(row, flight_rules_map), axis=1
    )

    # One last reset index
    metar_data_df.reset_index(inplace=True, drop=True)

    metar_data_df.dropna(
    subset=["time.dt", "density_altitude", "pressure_altitude", "altimeter_hpa"],
    inplace=True
    )

    print('Cleaning completed')

    return metar_data_df