# Free Flight Lab Weather-Related Departure Delay Predictor
## 1. Project and Content Description
The goal of this project is to generate departure delay probability predictions given the weather at the scheduled time of departure for a given route in the context of regularly scheduled airline operations.

The notebooks and scripts provided in this project are designed to compile flight and weather data, preprocess it for machine learning, define a machine learning pipeline, and train the selected machine learning algorithm for a binary classification of a given passenger flight departure as either "on-time" or "delayed". 

## 2. Dashboard
The project results were deployed on a streamlit app which can be accessed via [ffl-delay-predictor.streamlit.app/](https://ffl-delay-predictor.streamlit.app/)

![Dashboard Screenshot](/images/dashboard_screenshot.png)

## 3. Project status
This project was completed as a Minimum Viable Product / Proof of Concept and is no longer actively maintained. It is therefore open to forking or further development per the corresponding [MIT License and associated conditions](#license). Please refer to the [Feasibility & Roadmap section](#feasibility--roadmap) for recommended next steps.

## 4. Data and Budget
### Sources
The following APIs were chosen for this project due to their reliability and widespread industry usage. As such, they reflect state-of-the-art sources for flight information that is publically available and commercializable, albeit not free of cost:

- **[flightradar24 API](https://fr24api.flightradar24.com/docs/endpoints)** was used to collect specific flight identifiers corresponding to real flights at specific timestamps within the chosen date range of analysis.
- **[FlightAware API](https://www.flightaware.com/aeroapi/portal/documentation#overview)** was used to query the delay information corresponding to the flight identifiers that were collected using flightradar24.
- **[AVWX API](https://avwx.docs.apiary.io/#)** was used to collect the actual meteorological conditions as reported by the departure airport METARs at the scheduled time of departure. The corresponding AVWX parsing engine was used to parse the METAR data ([GitHub repo here](https://github.com/avwx-rest/avwx-engine)) in combination with a natively developed dataframe compiler.
- **[OAG](https://www.oag.com)** compiles (some) freely available route data which we used as a reference for compiling the routes of interest in our dataset.

The deployment of these sources is described in the [usage section](#usage).

### Budget
The project was completed on-time and on-budget in the context of a Data Science Capston project for Constructor Academy (Zürich) with the following resources:

- 2'000 CHF for APIs/data
- 480 development hours (3 data scientists, 4 full-time weeks)
- 5 support hours from our industry advisor
- 40 support hours from our technical advisors

## 5. Prior Work
This project was inspired by a weather-delay analysis dashboard developed under the [Airbus MONARK](https://acubed.airbus.com/projects/monark/) program. The link to the dashboard is provided [here](https://datascribe.shinyapps.io/MonaRk/) for reference (please be advised that it requires some time to load, please be patient).

## 6. Installation and Requirements
### Prerequisites
- Python 3.10.12 installed. You can download it from [python.org](https://www.python.org/downloads/release/python-31012/).
- `pip` should be installed and up to date (`python -m pip install --upgrade pip`).

### Setup Instructions

#### A. Create a Virtual Environment
It is recommended to create a virtual environment to manage dependencies.

##### Using `venv` (Standard)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
##### Using `conda` (Optional)
```bash
conda create --name my_project_env python=3.10.12
conda activate my_project_env
```

#### B. Install Dependencies
Ensure you have pip updated:

```bash
pip install --upgrade pip
```
Then install the dependencies from requirements.txt:

```bash
pip install -r env_requirements.txt
```

#### C. Verify Installation
Run the following command to ensure all dependencies are installed correctly:

```bash
python -c "import yaml, requests, numpy, pandas, sklearn, imblearn, shap, joblib, plotly, seaborn, matplotlib, folium, streamlit, avwx, IPython; print('All packages installed successfully!')"
```

## 7. Usage
This section outlines the methodology that you should follow to reproduce our results. Although we point to our notebooks and scripts specifically, this section was compiled with the aim that our results be reproducible if the general methodology is followed, rather than if our specific notebooks and scripts are deployed.

**Note on ICAO vs. IATA codes:** where relevant and available, it is recommended that ICAO identifiers are used for aircraft types, operators, flight-identifiers etc. due to their property of being unique in each attribute context. 

### 7.1. Compiling Flight Routes
- a) Compile the ICAO codes for origin and destination airports of the target routes (*origin_ICAO-destination_ICAO*) in an iterable format for the target API request parameterization.
  - **NOTE:** It is at this step that you can inadvertently introduce bias e.g. by selecting routes that are chronically delayed or which have unusually high on-time performance relative to industry standards. A well diversified route-group is highly recommended if performing a general analysis vs. a route-specific analysis. 

**Reference(s):**
- Sample [dataset](example_data/routes_by_region_2024_v3.csv) and the [notebook](notebooks/FIN_1_routes_by_region.ipynb) used to compile route data from OAG sources

### 7.2. Gathering Flightradar24 Data
- a) Define the date range for which you want to extract flight information
- b) Define the parameters for the [Historic Flight Positions Full](https://fr24api.flightradar24.com/docs/endpoints/overview#historic-flight-positions-full) flightradar24 API endpoint:
  - i) *timestamp*: the endpoint requires a specific UNIX timestamp per query
  - ii) *routes*: iterate through the routes compiled in [step 1](#1-compiling-flight-routes) for flight ID extraction corresponding to these routes
  - ii) *categories*: we defined **'P'** to limit extraction to flight IDs for commercial aircraft that carry passengers as their primary purpose 
  - iv) *operating_as*: permits you to limit extraction to target operators as defined by their unique airline ICAO code **(optional)**
  - v) *limit*: limit the number of results per query (if necessary, based on your budget)
- c) Define the querying logic to extract target flight IDs that match your target analysis
  - e.g. we queried once per day per route over our defined date range and randomized the single daily timestamp over the 24 hour period corresponding to the query date. As passenger flights are regularly scheduled, often over the year, our intent in querying a different time each day was to maximize the number of different operators and flights over our target date range, while limiting our query to 1 return per day per route. 

**Reference(s):**
- [flightradar24 extraction notebook](notebooks/FIN_2_Gathering_Flightradar24_Data.ipynb)

### 7.3. Gathering FlightAware Data 
- a) Compile the unique flight identifiers collected in [step 2](#2-gathering-flightradar24-data) into an iterable format
  - **Recommendation**: use the *callsign* attribute returned for each flight in your flightradar24 query
- b) Query the [get information for a historical flight](https://www.flightaware.com/aeroapi/portal/documentation#get-/history/flights/-ident-) FlightAware API endpoint using your unique flight identifiers

**Reference(s):**
- [FlightAware extraction notebook](notebooks/FIN_3_flightaware_data_extraction_by_FR24_callsign.ipynb)

### 7.4. Gathering AVWX Data
- a) Compile the ICAO codes and dates corresponding to the origin airports in the dataset you gathered in [step 3](#3-gathering-flightaware-data).
- b) Define the querying logic for the [Station History: Get Reporty by Station](https://avwxhistory.docs.apiary.io/#introduction/flight-path-routing) AVWX API endpoint.
  - e.g. we first queried based on the date, followed by the airport. This way we were able to save partial .csv files that contained only the METAR data for a given time frame.

**Reference(s):**
- [METAR extraction notebook](notebooks/FIN_4_Gathering_METAR_Data.ipynb)

### 7.5. Clean the METAR data and compile with the route data
- a) Compile the METAR data collected in [step 4](#4-gathering-avwx-data).
- b) Clean and prepare METAR data for Machine Learning based on your needs
  - i) Following the reference compilation notebook and meter cleaning script reflects our cleaning needs and logic, mostly based on one hot encoding. However, decisions were made regarding how to compile different Wx codes which may or may not reflect your needs or technical direction, the authors therefore encourage you to adapt the metar cleaning script based on your needs and domain knowledge. 
- c) Merge the METAR data with the route data based on your training criteria and research hypothesis
  - i) e.g. the model we used was trained on the actual weather conditions as reported by the airport-issued METAR at the **scheduled time of departure**. This was used in favor of the actual time of departure under the assumption that, if the weather caused the delay, it did so at the scheduled time of the departure, and not at the actual time of departure when weather conditions were presumably fit for flight.

**Reference(s):**
- [METAR compilation notebook](notebooks/FIN_5_Cleaning_METAR_Data.ipynb)
- [METAR cleaning script](notebooks/metar_cleaning.py)

**NOTE if you intend to use aviation weather forecasts (TAF) rather than reports (METAR):** although aviation weather forecasts (TAF) share key attributes with the aviation weather reports (METAR) we used in our methodology, you will need to adapt the dataframe compilation logic and code; the TAF formats are more variable and may not include all the same attributes.

### 7.6. Pre-Processing for Machine Learning
- a) Perform due-diligence cleaning such as dropping rows with missing values for attributes that are too risky to impute or which cannot otherwise be ignored.
- b) Create new features which could have predictive value (e.g. encoding weather conditions which satisfy airport-specific criteria for flight separation, or additional regional categories)
- c) Standardize your date/time fields and, if appropriate to your training needs and hypothesis, categorize them
- d) Delay class definition and creation
  - i) e.g. binary delay or multi-class delay
- e) Drop repeated attributes/features as well as attributes/features that are self-evidently useless
  - i) e.g. IATA vs. ICAO identifiers for the same root level attribute such as an operator/aircraft type are redundant
  - ii) e.g. the 'foresight_predictions_available' FlightAware binary is a self-evidently useless feature/attribute as it relates to the possibility of other information being available through a different endpoint in the FlightAware API
- f) Perform target leakage checks
  - i) e.g. 1) an arrival delay predictor may or may not have leakage from an attribute which includes the departure delay flag
  - ii) e.g. 2) a 'cancelled' attribute which was not removed is a source of leakage for delay prediction

**Reference(s):**
- [Pre-Processing for Machine Learning Notebook](notebooks/FIN_6_Pre-Processing_for_ML.ipynb) 

**NOTE on feature selection:** we treated formal feature selection as part of step [7.7 Machine Learning Training](#77-machine-learning-training)

### 7.7. Machine Learning Training
- a) Define and execute your train-test split
- b) Define/adapt your preprocessing pipelines 
  - i) note that many of the METAR features may already be one-hot encoded if you used the cleaning script referenced in the [METAR cleaning step](#75-clean-the-metar-data-and-compile-with-the-route-data) and can therefore be 'passed through'
- c) Establish a baseline
  - i) e.g. we performed a logistic regression with three different rebalancing methods: SMOTE, under-sampling, and class-weights
- d) Define your performance criteria based on your needs
  - i) e.g. we used the F-2 score as a performance criteria to favor recall over precision under the technical assumption that a false-negative (i.e. prediciting an actual delay as not delayed) is costlier to airline operations than predicting a false positive (predicing a delay when there is none). 
    - **NOTE**: performance reporting for this project was done on an F-1 score basis as it is more readily understandable to a wider audience than the F-2 score. In any case, whatever performance criteria you choose to optimize in your training should reflect your stakeholder needs.
- e) Experiment with different models
  - i) We used a combination of [TPOT](https://epistasislab.github.io/tpot/) for model selection and [Optuna](https://optuna.readthedocs.io/en/stable/) for tuning
- f) Understand what is driving predictions in your models with [explainability methods](#78-explainability)
- g) Further feature selection and engineering
- h) iterate from step b)

**Reference(s):**
- [Training for Machine Learning Notebook](notebooks/FIN_7_Machine_Learning_Training.ipynb) 

**NOTE on Feature Selection:** the reference notebook shows the 'end-state' of our iterative and explainability-driven feature selection process

**NOTE on Google Colab:** we used Google Colab for our Machine Learning modules and some of the Google Colab specific code requirements are therefore included in this section's notebook(s)

### 7.8. Explainability
- a) Employ SHAP methods (or otherwise) to understand what is driving predictions in your dataset-ML interaction
- b) Explore a parallel model that is trained on weather data only to better understand what weather elements may be of highest predictive value
- c) Explore a parallel model that is trained on operational data only to better understand what operational elements may be of highest predictive value

**Reference(s):**
- [Explainability Notebook](notebooks/FIN_8_Explainability.ipynb)

### 7.9. Dashboard
A notebook is provided as a partial example for prototyping the visuals we included in our dashboard. Please see [section 2. Dashboard](#2-dashboard) for our final dashboard deliverable. 

**Reference(s):**
- Dashboard graph example [notebook](notebooks/FIN_9_Dashboard_graphs.ipynb)
- Dashboard app [link](https://ffl-delay-predictor.streamlit.app/)

## 8. Observations, Feasibility & Roadmap
### Observations
- **The dataset we compiled inadvertently biases the algorithm to favor operational factors**
  - Running SHAP explainability algorithms on our predictions showed that the algorithm favors the features that match the input to a specific airport/route etc, even when these are absent in the training feature-set (i.e. weather only). As such, it is likely that our dataset's predictive signal has more to do with operators and airports, rather than weather input at scheduled time of departure. 
- **Delay prediction quality is better when ther are fewer operational variables in a route**
  - Although the aggregate prediction statistics for our algorithm are mid-range (F1-score in the 70% range), some routes showed predictive qualities as high as 90% and as low as 50%. Further exploration of route-specific results leads us to suggest that the development of operator-and-route-specific models may yield the levels of accuracy required for deployment in an airline Network Operations Center (NOC) context. 
- **Disruptive weather is less exciting than we thought**
  - Running the SHAP explainability algorithms on our predictions also showed that low meteorological visibility dominated the feature importance charts within the weather-feature subset of the algorithm input. This is likely due to the legal requirements that low visibility conditions impose on Air Traffic Control (ATC) for flight separation in LIFR conditions even in the absence of more notable disruptive weather events such as snow storms and icing conditions. 
- **Extreme aviation weather is rare and was ignored by our feature importance analysis tool**
  - Significant weather events (e.g. snow storms) were entirely absent in the feature importance charts despite the empirical evidence of how disruptive these events are to flight operations. This is likely because such events are relatively rare and short-lived in the context of 100'000+ flights a day, worldwide. Ensuring that such events are captured in the training dataset for an operator-and-route-specific model is therefore necessary to assess the feature importance related to these events, especially in routes that are more exposed to them. 

### Feasibility
Based on the [observations outlined above](#observations), the authors conclude that developing an operations-level prediction suite for weather-related delays is feasible. The authors therefore recommend the steps outlined in the [roadmap subsection below](#roadmap) as further work towards the definition of requirements for, and corresponding deployment of, a preliminary weather-related delay prediction suite design. 

### Roadmap
1. **Operator-and-Route-Specific Prediction Models** based on operator-and-route-specific data, would remove the variability that comes with the multiple operators and ground operations providers that are captured when aggregating routes from a single departure point, irrespective of operator. Developing such models would permit the calculation of a probability delay for a specific route within a single operational context. The predictions for each route can then be fed to the dynamic schedule/Gantt charts formats that are used for schedule planning in an airline's Network Operation Center (NOC).
2. **Weather Forecast vs. Report Comparison** is necessary in order to assess the quality of the key model input: the weather conditions forecasted at the scheduled time of departure. As the deployment of a prediction model in a real-world context would rely on aviation weather forecasts (TAFs), the assessment/characterization of historical TAF accuracy vs. actual METAR reported conditions for a given airport would be useful for dispatch decisions at an airline Network Operations Center (NOC) level.
3. **Quantified Delay Predictions** would give a more actionable output in the context of an airline Network Operations Center (NOC). Our model predicts a binary "on-time" or "delayed" classification; where a delay is defined as an actual gate-off time later than 15 minutes after the scheduled gate-off time (the industry standard). In this context, the prediciton makes no distinction between a 20 minute delay (which may have limited impact) and a 120 minute delay (which is presumably disruptive to an airline and its passengers). For simplicity, it is recommended that such a quantification be approached as a multi-class classification problem, rather than a full numerical regression problem. In such a case, the classification classes (e.g. minor delay, moderate delay, severe delay) should be defined in accordance to what is relevant to industry.  
4. **A Broader Dataset** that captures additional weather variables such as SIGMETs and winds-aloft would potentially permit the expansion of this methodology from departure delay predictions, to arrival delay predictions. 

## 9. Contributing
[Please see project status](#project-status).

## 10. Support
Please contact [the authors](#authors) in case support is required.

## 11. Acknowledgements 
The authors would like to give a special thanks to Mr. Kristjan Rognvaldsson who generously provided his invaluable industry expertise and technical guidance on a pro-bono basis. This project would not have been possible without him. 

We would also like to thank our advisors at the [Constructor Academy](https://academy.constructor.org/) in Zürich for their guidance in making this project a success.  

## 12. Sponsor
This project was completed through the generous sponsorship of the [Free Flight Lab](https://freeflightlab.org/).

## 13. Authors and Advisors
### Authors
- **Martina Wengle:** [LinkedIn](https://www.linkedin.com/in/martinawengle/)
- **Ralf Reuvers:** [LinkedIn](https://www.linkedin.com/in/ralf-reuvers-265b34282/)
- **René Marcel Falquier:** [GitHub](https://github.com/rmfalquier), [LinkedIn](www.linkedin.com/in/rmfalquier)

### Industry Advisor
- **Kristjan Rognvaldsson:** [LinkedIn](https://www.linkedin.com/in/kristjanrox/)

### Advisors
- **Albin Plathottathil:** [LinkedIn](https://www.linkedin.com/in/albin-plathottathil/)
- **Ekaterina Butyugina PhD:** [LinkedIn](https://www.linkedin.com/in/ekaterina-butyugina/)
- **Kunal Sharma PhD:** [LinkedIn](https://www.linkedin.com/in/drkunalsharma/)

## 14. License
This project is covered by an MIT license; please refer to the LICENSE file in this repository. 