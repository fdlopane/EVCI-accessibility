"""
Measuring supply and demand of electric vehicle charging infrastructure (EVCI) in London

Author Fulvio D. Lopane
Centre for Advanced Spatial Analysis

started coding: October 2024
"""

import pandas as pd
import geopandas as gpd
from config import *
import statsmodels
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import numpy as np
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from scipy.spatial import distance
from utils import *

# Create the EVCI-year column in EVCI data set
if not os.path.exists(generated["EVCI-London"]):
    # Import EVCI counts at lsoa level
    EVCI = pd.read_csv(inputs["EVCI-points-raw"])
    # Extract the year from the dateCreate column
    EVCI["EVCIyear"] = EVCI["dateCreate"].str[6:10]
    # Save the data
    EVCI.to_csv(generated["EVCI-London"], index=False)

# Import EVCI counts at lsoa level
EVCI = pd.read_csv(inputs["EVCI-LSOA"])

# Import EV licensing counts at lsoa level
# Note 338 out of 4994 LSOAs have null values
EV_counts = pd. read_csv(inputs["EV-counts-LSOA"])

# Deal with null values is EV_counts --> imputed value of 2.5 for the following reason
# Data source: https://www.gov.uk/government/statistical-data-sets/vehicle-licensing-statistics-data-files
'''
In order to keep these files to a reasonable size,
small areas which have never had 5 or more vehicles in scope are excluded from these datasets,
with those vehicles combined into the Miscellaneous row. This means a number of LSOAs will be missing from the file.
'''
EV_counts.fillna(2.5, inplace=True)

# Merge the datafames for regression analysis
analysis_df = EVCI.merge(EV_counts, on='LSOA21CD')
# Rename and drop duplicate columns
# NOTE delete ALL dashes (-), spaces, and underscores (_) from column names
analysis_df.drop(columns=['LSOA21NM_y', 'GlobalID_y'], inplace=True)

analysis_df.rename(columns={'LSOA21NM_x': 'LSOA21NM',
                            'GlobalID_x': 'GlobalID',
                            'EVCI-2024': 'EVCI2024',
                            'EVCI-2023': 'EVCI2023',
                            'EVCI-2022': 'EVCI2022',
                            'EVCI-2021': 'EVCI2021',
                            'EVCI-2020': 'EVCI2020',
                            'EVCI-2019': 'EVCI2019',
                            'EVCI-2018': 'EVCI2018',
                            'EVCI-2017': 'EVCI2017',
                            'EVCI-2015': 'EVCI2015',
                            'EVCI-2013': 'EVCI2013',
                            'EVCI-2012': 'EVCI2012',
                            '_Fuel': 'Fuel',
                            '_Keepershi': 'Keepership',
                            '_2024 Q2': 'y2024Q2',
                            '_2024 Q1': 'y2024Q1',
                            '_2023 Q4': 'y2023Q4',
                            '_2023 Q3': 'y2023Q3',
                            '_2023 Q2': 'y2023Q2',
                            '_2023 Q1': 'y2023Q1',
                            '_2022 Q4': 'y2022Q4',
                            '_2022 Q3': 'y2022Q3',
                            '_2022 Q2': 'y2022Q2',
                            '_2022 Q1': 'y2022Q1',
                            '_2021 Q4': 'y2021Q4',
                            '_2021 Q3': 'y2021Q3',
                            '_2021 Q2': 'y2021Q2',
                            '_2021 Q1': 'y2021Q1',
                            '_2020 Q4': 'y2020Q4',
                            '_2020 Q3': 'y2020Q3',
                            '_2020 Q2': 'y2020Q2',
                            '_2020 Q1': 'y2020Q1',
                            '_2019 Q4': 'y2019Q4',
                            '_2019 Q3': 'y2019Q3',
                            '_2019 Q2': 'y2019Q2',
                            '_2019 Q1': 'y2019Q1',
                            '_2018 Q4': 'y2018Q4',
                            '_2018 Q3': 'y2018Q3',
                            '_2018 Q2': 'y2018Q2',
                            '_2018 Q1': 'y2018Q1',
                            '_2017 Q4': 'y2017Q4',
                            '_2017 Q3': 'y2017Q3',
                            '_2017 Q2': 'y2017Q2',
                            '_2017 Q1': 'y2017Q1',
                            '_2016 Q4': 'y2016Q4',
                            '_2016 Q3': 'y2016Q3',
                            '_2016 Q2': 'y2016Q2',
                            '_2016 Q1': 'y2016Q1',
                            '_2015 Q4': 'y2015Q4',
                            '_2015 Q3': 'y2015Q3',
                            '_2015 Q2': 'y2015Q2',
                            '_2015 Q1': 'y2015Q1',
                            '_2014 Q4': 'y2014Q4',
                            '_2014 Q3': 'y2014Q3',
                            '_2014 Q2': 'y2014Q2',
                            '_2014 Q1': 'y2014Q1',
                            '_2013 Q4': 'y2013Q4',
                            '_2013 Q3': 'y2013Q3',
                            '_2013 Q2': 'y2013Q2',
                            '_2013 Q1': 'y2013Q1',
                            '_2012 Q4': 'y2012Q4',
                            '_2012 Q3': 'y2012Q3',
                            '_2012 Q2': 'y2012Q2',
                            '_2012 Q1': 'y2012Q1',
                            '_2011 Q4': 'y2011Q4',
                            '_2021 Q1': 'y2021Q1'}, inplace=True)

########################################################################################################################
# Remove EV licensing outliers (remove those LSOAs with licensing > lic_threshold)
lic_threshold = 1000
analysis_df = analysis_df.drop(analysis_df[analysis_df.y2024Q2 > lic_threshold].index)

########################################################################################################################
# Measure correlation between supply and demand
SD_flag = False

supply_var = "EVCI2024"
demand_var = "y2024Q2"

if SD_flag == True:
    calculate_corr_matrix_2_var(analysis_df, supply_var, demand_var)

########################################################################################################################
# 2021 analysis

analysis_2021 = analysis_df[["LSOA21CD", "LSOA21NM", "EVCI2021", "y2021Q4"]]

# London LSOA codes
Londn_LSOAs = pd.read_csv(inputs["London_LSOAs"])
London_LSOA_codes = Londn_LSOAs["LSOA21CD"].tolist()

# Filter out London from GB mean house prices
if not os.path.exists(generated["Mean_house_prices_London"]):
    HP_GB = pd.read_csv(inputs["Mean_house_prices_GB"])
    HP_London = HP_GB[HP_GB["LSOA code"].isin(London_LSOA_codes)]
    HP_London.rename(columns={"LSOA code": "LSOA21CD"}, inplace=True)
    HP_London.to_csv(generated["Mean_house_prices_London"], index=False)

# Fitler out London from GB median house prices
if not os.path.exists(generated["Median_house_prices_London"]):
    MP_GB = pd.read_csv(inputs["Median_house_prices_GB"])
    MP_GB = MP_GB[["Local authority code", "Local authority name", "LSOA code", "LSOA name", "Year ending Dec 2021"]]
    MP_London = MP_GB[MP_GB["LSOA code"].isin(London_LSOA_codes)]
    MP_London.rename(columns={"LSOA code": "LSOA21CD"}, inplace=True)
    MP_London.to_csv(generated["Median_house_prices_London"], index=False)

Mean_house_prices_London = pd.read_csv(generated["Mean_house_prices_London"])
Median_house_prices_London = pd.read_csv(generated["Median_house_prices_London"])

# Deprivation index data
HH_deprivation = pd.read_csv(inputs["HH_deprivation_2021"])
HH_deprivation.rename(columns={"LSOA code": "LSOA21CD",
                               "All Households": "HH_number",
                               "deprived in: no dimensions": "D0",
                               "1 dimension": "D1",
                               "2 dimensions": "D2",
                               "3 dimensions": "D3",
                               "4 dimensions": "D4"}, inplace=True)

# Approximate social grade (ASG) data (extract London from GB data):
if not os.path.exists(generated["ASG_London"]):
    ASG_GB = pd.read_csv(inputs["ASG_GB"])
    ASG_GB.rename(columns={"2021 super output area - lower layer": "LSOA21NM",
                        "mnemonic": "LSOA21CD",
                        "AB Higher and intermediate managerial/administrative/professional occupations": "ASG_AB",
                        "C1 Supervisory, clerical and junior managerial/administrative/professional occupations": "ASG_C1",
                        "C2 Skilled manual occupations": "ASG_C2",
                        "DE Semi-skilled and unskilled manual occupations; unemployed and lowest grade occupations": "ASG_DE"},
               inplace=True)
    ASG_London = ASG_GB[ASG_GB["LSOA21CD"].isin(London_LSOA_codes)]
    ASG_London.to_csv(generated["ASG_London"], index=False)

ASG_London = pd.read_csv(generated["ASG_London"])

# Get rid of double entries:
ASG_London = ASG_London[ASG_London.duplicated(["LSOA21CD"])]
ASG_London.reset_index()

print()
print("###############################################################################################################")
print("WARNING: London has ", len(London_LSOA_codes), " LSOAs.")
print("House prices are available for ", len(Mean_house_prices_London), " LSOAs.")
print("ASG available for ", len(ASG_London), "LSOAs.")
print("Deprivation data available for ", len(HH_deprivation), "LSOAs.")
print("EVCI supply available for ", len(EVCI), "LSOAs.")
print("EV licensing available for ", len(EV_counts), "LSOAs.")
print("###############################################################################################################")
print()

# Merge the house prices, deprivation index, and ASG to the supply & demand dataframe
# MEMO: use median house prices (not mean) for the analysis

# Only keep relevant columns:
Median_house_prices_London = Median_house_prices_London[["LSOA21CD", "Year ending Dec 2021"]]
Median_house_prices_London.rename(columns={"Year ending Dec 2021": "Med_HP_2021"}, inplace=True)
# Remove commas and ":" from the median house prices
Median_house_prices_London["Med_HP_2021"] = Median_house_prices_London["Med_HP_2021"].str.replace(",", "")
Median_house_prices_London["Med_HP_2021"] = Median_house_prices_London["Med_HP_2021"].str.replace(":", "")
# Drop the rows with "" in the median house prices
Median_house_prices_London = Median_house_prices_London[Median_house_prices_London["Med_HP_2021"] != ""]
# Convert the median house prices to float
Median_house_prices_London["Med_HP_2021"] = Median_house_prices_London["Med_HP_2021"].astype(float)

ASG_London = ASG_London[["LSOA21CD", "ASG_AB", "ASG_C1", "ASG_C2", "ASG_DE"]]

HH_deprivation = HH_deprivation[["LSOA21CD", "HH_number", "D0", "D1", "D2", "D3", "D4"]]

# Merge the dataframes
analysis_2021 = analysis_2021.merge(Median_house_prices_London, on="LSOA21CD", how="outer")
analysis_2021 = analysis_2021.merge(ASG_London, on="LSOA21CD", how="outer")
analysis_2021 = analysis_2021.merge(HH_deprivation, on="LSOA21CD", how="outer")

# Create an accessibility column by deviding the n of chargers by the number of EV
analysis_2021["accessibility"] = analysis_2021["EVCI2021"] / analysis_2021["y2021Q4"]

# Remove LSOA codes and names, and HH number
#analysis_2021.drop(columns=["LSOA21CD", "LSOA21NM", "HH_number", "EVCI2021", "y2021Q4"], inplace=True)
#analysis_2021.drop(columns=["LSOA21CD", "LSOA21NM", "HH_number"], inplace=True)
analysis_2021.drop(columns=["LSOA21NM", "HH_number"], inplace=True)

# Correlation matrix
CSCA_2021_flag = False
if CSCA_2021_flag == True:
    plt_and_save_corr_matrix(analysis_2021, outputs["correlation_matrix_2021"])

# OLS analysis 2021
OLS_2021_flag = False
if OLS_2021_flag == True:

    # OLS for SUPPLY
    dependent_variable = "EVCI2021"
    independent_variables = ["y2021Q4", "Med_HP_2021", "ASG_AB", "ASG_C1", "ASG_C2", "ASG_DE", "D0", "D1", "D2", "D3", "D4"]
    OLS_analysis(analysis_2021, dependent_variable, independent_variables)

    # OLS for DEMAND
    dependent_variable = "y2021Q4"
    independent_variables = ["EVCI2021", "Med_HP_2021", "ASG_AB", "ASG_C1", "ASG_C2", "ASG_DE", "D0", "D1", "D2", "D3", "D4"]
    OLS_analysis(analysis_2021, dependent_variable, independent_variables)

    # OLS for ACCESSIBILITY
    dependent_variable = "accessibility"
    independent_variables = ["EVCI2021", "Med_HP_2021", "ASG_AB", "ASG_C1", "ASG_C2", "ASG_DE", "D0", "D1", "D2", "D3", "D4"]
    OLS_analysis(analysis_2021, dependent_variable, independent_variables)

# Geographically Weighted Regression (GWR) analysis 2021

GWR_flag = True
if GWR_flag == True:
    # import the London centroids shapefile with geopandas
    London_LSOA_centroids = gpd.read_file(inputs["London_LSOA_centroids"])
    # Extract the coordinates of the centroids
    London_LSOA_centroids["x"] = London_LSOA_centroids.centroid.x
    London_LSOA_centroids["y"] = London_LSOA_centroids.centroid.y

    # Only keep the relevant columns
    London_LSOA_centroids = London_LSOA_centroids[["LSOA21CD", "x", "y"]]

    # Merge the centroids with the analysis_2021 dataframe
    analysis_2021 = analysis_2021.merge(London_LSOA_centroids, on="LSOA21CD", how="outer")
    #analysis_2021.drop(columns=["LSOA21CD"], inplace=True)

    # Remove NaNs
    analysis_2021 = analysis_2021.dropna()

    # Step 1: Prepare spatial coordinates and data
    u = analysis_2021["x"]
    v = analysis_2021["y"]
    coords = list(zip(u,v))  # Spatial coordinates
    X = analysis_2021[["Med_HP_2021"]].values  # Independent variables
    Y = analysis_2021["accessibility"].values.reshape(-1, 1)  # Dependent variable reshaped for GWR

    # Step 2: Select the optimal bandwidth
    selector = Sel_BW(coords, Y, X)
    bandwidth = selector.search()
    print(f"Optimal Bandwidth: {bandwidth}")

    # Step 3: Fit the GWR model
    gwr_results = GWR(coords, Y, X, bandwidth).fit()

    # Step 4: Inspect results
    print(gwr_results.summary())

    # Step 5: Extract local coefficients
    #local_coefficients = gwr_results.params
    #print("Local Coefficients:\n", local_coefficients)


    London_LSOA_polygons = gpd.read_file(inputs["London_LSOA_polygons"])
    # Now use the polygons geometry for the analysis_2021 dataframe
    analysis_2021 = analysis_2021.merge(London_LSOA_polygons, on="LSOA21CD", how="outer")
    # Now turn the analysis_2021 dataframe into a geodataframe
    analysis_2021 = gpd.GeoDataFrame(analysis_2021)

    analysis_2021.plot()
    plt.show()

# Link to GWR tutorial (follow for visualisation
# https://deepnote.com/app/carlos-mendez/PYTHON-GWR-and-MGWR-71dd8ba9-a3ea-4d28-9b20-41cc8a282b7a

'''
    analysis_2021['gwr_R2'] = gwr_results.localR2

    # Step 6: Visualize local R^2
    fig, ax = plt.subplots(figsize=(6, 6))
    analysis_2021.plot(column='gwr_R2', cmap='coolwarm', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
             legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=ax)
    ax.set_title('Local R2', fontsize=12)
    ax.axis("off")
    # plt.savefig('myMap.png',dpi=150, bbox_inches='tight')
    plt.show()

'''

'''
    local_r2 = gwr_results.localR2
    plt.scatter(analysis_2021['x'], analysis_2021['y'], c=local_r2, cmap='viridis', s=50)
    plt.colorbar(label='Local R²')
    plt.title("Geographically Weighted Regression: Local R²")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
'''