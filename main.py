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
# 2021 vs 2024 analysis

analysis_21_24 = analysis_df[["LSOA21CD", "LSOA21NM", "EVCI2021", "y2021Q4", "EVCI2024", "y2024Q2"]]

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
    MP_GB = MP_GB[["Local authority code", "Local authority name", "LSOA code", "LSOA name", "Year ending Dec 2021", "Year ending Mar 2023"]]
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
Median_house_prices_London = Median_house_prices_London[["LSOA21CD", "Year ending Dec 2021", "Year ending Mar 2023"]]
Median_house_prices_London.rename(columns={"Year ending Dec 2021": "Med_HP_2021",
                                           "Year ending Mar 2023": "Med_HP_2023"}, inplace=True)
# Remove commas and ":" from the median house prices
Median_house_prices_London["Med_HP_2021"] = Median_house_prices_London["Med_HP_2021"].str.replace(",", "")
Median_house_prices_London["Med_HP_2021"] = Median_house_prices_London["Med_HP_2021"].str.replace(":", "")
Median_house_prices_London["Med_HP_2023"] = Median_house_prices_London["Med_HP_2021"].str.replace(",", "")
Median_house_prices_London["Med_HP_2023"] = Median_house_prices_London["Med_HP_2021"].str.replace(":", "")

# Drop the rows with "" in the median house prices
Median_house_prices_London = Median_house_prices_London[Median_house_prices_London["Med_HP_2021"] != ""]
Median_house_prices_London = Median_house_prices_London[Median_house_prices_London["Med_HP_2023"] != ""]
# Convert the median house prices to float
Median_house_prices_London["Med_HP_2021"] = Median_house_prices_London["Med_HP_2021"].astype(float)
Median_house_prices_London["Med_HP_2023"] = Median_house_prices_London["Med_HP_2023"].astype(float)

ASG_London = ASG_London[["LSOA21CD", "ASG_AB", "ASG_C1", "ASG_C2", "ASG_DE"]]

HH_deprivation = HH_deprivation[["LSOA21CD", "HH_number", "D0", "D1", "D2", "D3", "D4"]]

# Merge the dataframes
analysis_21_24 = analysis_21_24.merge(Median_house_prices_London, on="LSOA21CD", how="outer")
analysis_21_24 = analysis_21_24.merge(ASG_London, on="LSOA21CD", how="outer")
analysis_21_24 = analysis_21_24.merge(HH_deprivation, on="LSOA21CD", how="outer")

# Add population density
Population_density_GB = pd.read_csv(inputs["Population_density_GB"])
Population_density_London = Population_density_GB[Population_density_GB["LSOA21CD"].isin(London_LSOA_codes)]
Population_density_London = Population_density_London[["LSOA21CD", "People per Sq Km"]]
Population_density_London.rename(columns={"People per Sq Km": "Pop_density"}, inplace=True)
# Remove commes "," in the field to turn the Population density to float
Population_density_London["Pop_density"] = Population_density_London["Pop_density"].str.replace(",", "")
Population_density_London["Pop_density"] = Population_density_London["Pop_density"].astype(float)
analysis_21_24 = analysis_21_24.merge(Population_density_London, on="LSOA21CD", how="outer")

# Add number of cars per Household (HH)
Vehicle_ownership_London_2021 = pd.read_csv(inputs["Vehicle_ownership_London_2021"])
Vehicle_ownership_London_2021.rename(columns={"Lower layer Super Output Areas Code": "LSOA21CD",
                                              "Lower layer Super Output Areas": "LSOA21NM",
                                              "Car or van availability (5 categories) Code": "Car_aval_code",
                                              "Car or van availability (5 categories)": "Car_aval_category",
                                              "Observation": "n_of_HH"}, inplace=True)

# Restructure the dataframe to have one LSOA in every row, and creating new columns for each car availability category
Vehicle_ownership_London_2021 = Vehicle_ownership_London_2021.pivot(index="LSOA21CD", columns="Car_aval_category", values="n_of_HH")
Vehicle_ownership_London_2021.reset_index(inplace=True)
Vehicle_ownership_London_2021.columns.name = None
Vehicle_ownership_London_2021.rename(columns={"No cars or vans in household": "HH_cars_0",
                                              "1 car or van in household": "HH_cars_1",
                                              "2 cars or vans in household": "HH_cars_2",
                                              "3 or more cars or vans in household": "HH_cars_3+"}, inplace=True)
Vehicle_ownership_London_2021 = Vehicle_ownership_London_2021[['LSOA21CD', 'HH_cars_0', 'HH_cars_1', 'HH_cars_2', 'HH_cars_3+']]

# Merge the vehicle ownership data
analysis_21_24 = analysis_21_24.merge(Vehicle_ownership_London_2021, on="LSOA21CD", how="outer")

# Add house tenure
House_tenure_London_2021 = pd.read_csv(inputs["House_tenure_London_2021"])
# Rename columns
House_tenure_London_2021.rename(columns={"Lower layer Super Output Areas Code": "LSOA21CD",
                                              "Lower layer Super Output Areas": "LSOA21NM",
                                              "Tenure of household (9 categories) Code": "HH_tenure_code",
                                              "Tenure of household (9 categories)": "HH_tenure_category",
                                              "Observation": "n_of_HH"}, inplace=True)

# Restructure the dataframe to have one LSOA in every row, and creating new columns for each tenure of household category
House_tenure_London_2021 = House_tenure_London_2021.pivot(index="LSOA21CD", columns="HH_tenure_category", values="n_of_HH")
House_tenure_London_2021.reset_index(inplace=True)
House_tenure_London_2021.columns.name = None
# Rename columns
House_tenure_London_2021.rename(columns={"Owned: Owns outright": "HHT_owned_outright",
                                         "Owned: Owns with a mortgage or loan": "HHT_owned_mortgage",
                                         "Shared ownership: Shared ownership": "HHT_shared_ownership",
                                         "Private rented: Other private rented": "HHT_rented_other",
                                         "Private rented: Private landlord or letting agency": "HHT_rented_private",
                                         "Social rented: Other social rented": "HHT_rented_social",
                                         "Lives rent free": "HHT_rent_free",
                                         "Social rented: Rents from council or Local Authority": "HHT_rented_social"}, inplace=True)

House_tenure_London_2021 = House_tenure_London_2021[["LSOA21CD", "HHT_rent_free", "HHT_owned_outright",
                                                     "HHT_owned_mortgage", "HHT_rented_other", "HHT_rented_private",
                                                     "HHT_shared_ownership", "HHT_rented_social", "HHT_rented_social"]]

# Merge the house tenure data
analysis_21_24 = analysis_21_24.merge(House_tenure_London_2021, on="LSOA21CD", how="outer")

# Add accommodation type
Accommodation_type_London_2021 = pd.read_csv(inputs["Accommodation_type_London_2021"])
# Rename columns
Accommodation_type_London_2021.rename(columns={"Lower layer Super Output Areas Code": "LSOA21CD",
                                                "Lower layer Super Output Areas": "LSOA21NM",
                                                "Accommodation type (8 categories) Code": "Acc_type_code",
                                                "Accommodation type (8 categories)": "Acc_type_category",
                                                "Observation": "n_of_HH"}, inplace=True)
# Restructure the dataframe to have one LSOA in every row, and creating new columns for each accommodation type category
Accommodation_type_London_2021 = Accommodation_type_London_2021.pivot(index="LSOA21CD", columns="Acc_type_category", values="n_of_HH")
Accommodation_type_London_2021.reset_index(inplace=True)
Accommodation_type_London_2021.columns.name = None
# Rename columns
Accommodation_type_London_2021.rename(columns={"Detached": "Acc_detached",
                                               "A caravan or other mobile or temporary structure": "Acc_caravan",
                                               "In a commercial building, for example, in an office building, hotel or over a shop": "Acc_commercial",
                                               "In a purpose-built block of flats or tenement": "Acc_flat",
                                               "Part of a converted or shared house, including bedsits": "Acc_converted_or_shared",
                                               "Part of another converted building, for example, former school, church or warehouse": "Acc_converted_other",
                                               "Semi-detached": "Acc_semidetached",
                                               "Terraced": "Acc_terraced"}, inplace=True)

# Merge the accommodation type data
analysis_21_24 = analysis_21_24.merge(Accommodation_type_London_2021, on="LSOA21CD", how="outer")

# Create an accessibility column by dividing the n of chargers by the number of EV
analysis_21_24["accessibility_21"] = analysis_21_24["EVCI2021"] / analysis_21_24["y2021Q4"]
analysis_21_24["accessibility_24"] = analysis_21_24["EVCI2024"] / analysis_21_24["y2024Q2"]
analysis_21_24["acc_diff_24_21"] = analysis_21_24["accessibility_24"] / analysis_21_24["accessibility_21"]

# Remove columns
#analysis_21_24.drop(columns=["LSOA21CD", "LSOA21NM", "HH_number", "EVCI2021", "y2021Q4"], inplace=True)
#analysis_21_24.drop(columns=["LSOA21CD", "LSOA21NM", "HH_number"], inplace=True)
#analysis_21_24.drop(columns=["LSOA21NM", "HH_number"], inplace=True)

# Correlation matrix
CSCA_2021_2024_flag = False
if CSCA_2021_2024_flag == True:
    plt_and_save_corr_matrix(analysis_21_24, outputs["correlation_matrix_2021_2024"])

# print data type of each column
print(analysis_21_24.dtypes)
print(analysis_21_24.columns)

# OLS analysis
OLS_2021_flag = True
if OLS_2021_flag == True:

    # TODO: take the Log of house prices (and other big numbers to avoid coefficients with many zeros)

    # OLS for SUPPLY 2021
    dependent_variable = "EVCI2021"
    independent_variables = ["y2021Q4",                 # EV licensing 2021
                             "Med_HP_2021",             # Median house prices 2021 (December)
                             "ASG_AB",                  # Approx social grade (higher and intermediate occ.)
                             "ASG_C1",                  # Approx social grade (Supervisory and junior managerial  occ.)
                             "ASG_C2",                  # Approx social grade (Skilled manual occ.)
                             "ASG_DE",                  # Approx social grade (Semi-skilled, unempl., lowest grade occ.)
                             "D0",                      # Deprivation index 0 (no dimensions)
                             "D1",                      # Deprivation index 1 (1 dimension)
                             "D2",                      # Deprivation index 2 (2 dimensions)
                             "D3",                      # Deprivation index 3 (3 dimensions)
                             "D4",                      # Deprivation index 4 (4 dimensions)
                             "Pop_density",             # Population density
                             "HH_cars_0",               # N of HH with 0 cars
                             "HH_cars_1",               # N of HH with 1 car
                             "HH_cars_2",               # N of HH with 2 cars
                             "HH_cars_3+",              # N of HH with 3+ cars
                             "HHT_rent_free",           # N of HH living rent-free
                             "HHT_owned_outright",      # N of HH owning outright
                             "HHT_owned_mortgage",      # N of HH owning with mortgage
                             "HHT_rented_other",        # N of HH renting from other private landlords
                             "HHT_rented_private",      # N of HH renting from private landlords
                             "HHT_shared_ownership",    # N of HH in shared ownership
                             "HHT_rented_social",       # N of HH renting from social landlords
                             "Acc_detached",            # N of HH living in detached houses
                             "Acc_caravan",             # N of HH living in caravans
                             "Acc_commercial",          # N of HH living in commercial buildings
                             "Acc_flat",                # N of HH living in flats
                             "Acc_converted_or_shared", # N of HH living in converted or shared houses
                             "Acc_converted_other",     # N of HH living in other converted buildings
                             "Acc_semidetached",        # N of HH living in semi-detached houses
                             "Acc_terraced"]            # N of HH living in terraced houses

    supply_summary_table_21 = OLS_analysis(analysis_21_24, dependent_variable, independent_variables)
    # save the summary table
    supply_summary_table_21.to_csv(outputs["OLS_supply_2021"], index=False)

    # __________________________________________________________________________________________________________________
    # OLS for DEMAND 2021
    dependent_variable = "y2021Q4"
    independent_variables = ["EVCI2021",                # EVCI 2021
                             "Med_HP_2021",             # Median house prices 2021 (December)
                             "ASG_AB",                  # Approx social grade (higher and intermediate occ.)
                             "ASG_C1",                  # Approx social grade (Supervisory and junior managerial  occ.)
                             "ASG_C2",                  # Approx social grade (Skilled manual occ.)
                             "ASG_DE",                  # Approx social grade (Semi-skilled, unempl., lowest grade occ.)
                             "D0",                      # Deprivation index 0 (no dimensions)
                             "D1",                      # Deprivation index 1 (1 dimension)
                             "D2",                      # Deprivation index 2 (2 dimensions)
                             "D3",                      # Deprivation index 3 (3 dimensions)
                             "D4",                      # Deprivation index 4 (4 dimensions)
                             "Pop_density",             # Population density
                             "HH_cars_0",               # N of HH with 0 cars
                             "HH_cars_1",               # N of HH with 1 car
                             "HH_cars_2",               # N of HH with 2 cars
                             "HH_cars_3+",              # N of HH with 3+ cars
                             "HHT_rent_free",           # N of HH living rent-free
                             "HHT_owned_outright",      # N of HH owning outright
                             "HHT_owned_mortgage",      # N of HH owning with mortgage
                             "HHT_rented_other",        # N of HH renting from other private landlords
                             "HHT_rented_private",      # N of HH renting from private landlords
                             "HHT_shared_ownership",    # N of HH in shared ownership
                             "HHT_rented_social",       # N of HH renting from social landlords
                             "Acc_detached",            # N of HH living in detached houses
                             "Acc_caravan",             # N of HH living in caravans
                             "Acc_commercial",          # N of HH living in commercial buildings
                             "Acc_flat",                # N of HH living in flats
                             "Acc_converted_or_shared", # N of HH living in converted or shared houses
                             "Acc_converted_other",     # N of HH living in other converted buildings
                             "Acc_semidetached",        # N of HH living in semi-detached houses
                             "Acc_terraced"]            # N of HH living in terraced houses

    demand_summary_table_21 = OLS_analysis(analysis_21_24, dependent_variable, independent_variables)
    # save the summary table
    demand_summary_table_21.to_csv(outputs["OLS_demand_2021"], index=False)

    # __________________________________________________________________________________________________________________
    # OLS for ACCESSIBILITY 2021
    dependent_variable = "accessibility_21"
    independent_variables = ["Med_HP_2021",             # Median house prices 2021 (December)
                             "ASG_AB",                  # Approx social grade (higher and intermediate occ.)
                             "ASG_C1",                  # Approx social grade (Supervisory and junior managerial  occ.)
                             "ASG_C2",                  # Approx social grade (Skilled manual occ.)
                             "ASG_DE",                  # Approx social grade (Semi-skilled, unempl., lowest grade occ.)
                             "D0",                      # Deprivation index 0 (no dimensions)
                             "D1",                      # Deprivation index 1 (1 dimension)
                             "D2",                      # Deprivation index 2 (2 dimensions)
                             "D3",                      # Deprivation index 3 (3 dimensions)
                             "D4",                      # Deprivation index 4 (4 dimensions)
                             "Pop_density",             # Population density
                             "HH_cars_0",               # N of HH with 0 cars
                             "HH_cars_1",               # N of HH with 1 car
                             "HH_cars_2",               # N of HH with 2 cars
                             "HH_cars_3+",              # N of HH with 3+ cars
                             "HHT_rent_free",           # N of HH living rent-free
                             "HHT_owned_outright",      # N of HH owning outright
                             "HHT_owned_mortgage",      # N of HH owning with mortgage
                             "HHT_rented_other",        # N of HH renting from other private landlords
                             "HHT_rented_private",      # N of HH renting from private landlords
                             "HHT_shared_ownership",    # N of HH in shared ownership
                             "HHT_rented_social",       # N of HH renting from social landlords
                             "Acc_detached",            # N of HH living in detached houses
                             "Acc_caravan",             # N of HH living in caravans
                             "Acc_commercial",          # N of HH living in commercial buildings
                             "Acc_flat",                # N of HH living in flats
                             "Acc_converted_or_shared", # N of HH living in converted or shared houses
                             "Acc_converted_other",     # N of HH living in other converted buildings
                             "Acc_semidetached",        # N of HH living in semi-detached houses
                             "Acc_terraced"]            # N of HH living in terraced houses

    accessibility_summary_table_21 = OLS_analysis(analysis_21_24, dependent_variable, independent_variables)
    # save the summary table
    accessibility_summary_table_21.to_csv(outputs["OLS_accessibility_2021"], index=False)

    # __________________________________________________________________________________________________________________
    # OLS for SUPPLY 2024
    dependent_variable = "EVCI2024"
    independent_variables = ["y2024Q2",                 # EV licensing 2024 (June)
                             "Med_HP_2023",             # Median house prices 2023 (March)
                             "ASG_AB",                  # Approx social grade (higher and intermediate occ.)
                             "ASG_C1",                  # Approx social grade (Supervisory and junior managerial  occ.)
                             "ASG_C2",                  # Approx social grade (Skilled manual occ.)
                             "ASG_DE",                  # Approx social grade (Semi-skilled, unempl., lowest grade occ.)
                             "D0",                      # Deprivation index 0 (no dimensions)
                             "D1",                      # Deprivation index 1 (1 dimension)
                             "D2",                      # Deprivation index 2 (2 dimensions)
                             "D3",                      # Deprivation index 3 (3 dimensions)
                             "D4",                      # Deprivation index 4 (4 dimensions)
                             "Pop_density",             # Population density
                             "HH_cars_0",               # N of HH with 0 cars
                             "HH_cars_1",               # N of HH with 1 car
                             "HH_cars_2",               # N of HH with 2 cars
                             "HH_cars_3+",              # N of HH with 3+ cars
                             "HHT_rent_free",           # N of HH living rent-free
                             "HHT_owned_outright",      # N of HH owning outright
                             "HHT_owned_mortgage",      # N of HH owning with mortgage
                             "HHT_rented_other",        # N of HH renting from other private landlords
                             "HHT_rented_private",      # N of HH renting from private landlords
                             "HHT_shared_ownership",    # N of HH in shared ownership
                             "HHT_rented_social",       # N of HH renting from social landlords
                             "Acc_detached",            # N of HH living in detached houses
                             "Acc_caravan",             # N of HH living in caravans
                             "Acc_commercial",          # N of HH living in commercial buildings
                             "Acc_flat",                # N of HH living in flats
                             "Acc_converted_or_shared", # N of HH living in converted or shared houses
                             "Acc_converted_other",     # N of HH living in other converted buildings
                             "Acc_semidetached",        # N of HH living in semi-detached houses
                             "Acc_terraced"]            # N of HH living in terraced houses

    supply_summary_table_24 = OLS_analysis(analysis_21_24, dependent_variable, independent_variables)

    # save the summary table
    supply_summary_table_24.to_csv(outputs["OLS_supply_2024"], index=False)

    # __________________________________________________________________________________________________________________
    # OLS for DEMAND 2024
    dependent_variable = "y2024Q2"
    independent_variables = ["EVCI2024",                # EVCI 2024
                             "Med_HP_2023",  # Median house prices 2023 (March)
                             "ASG_AB",  # Approx social grade (higher and intermediate occ.)
                             "ASG_C1",  # Approx social grade (Supervisory and junior managerial  occ.)
                             "ASG_C2",  # Approx social grade (Skilled manual occ.)
                             "ASG_DE",  # Approx social grade (Semi-skilled, unempl., lowest grade occ.)
                             "D0",  # Deprivation index 0 (no dimensions)
                             "D1",  # Deprivation index 1 (1 dimension)
                             "D2",  # Deprivation index 2 (2 dimensions)
                             "D3",  # Deprivation index 3 (3 dimensions)
                             "D4",  # Deprivation index 4 (4 dimensions)
                             "Pop_density",  # Population density
                             "HH_cars_0",  # N of HH with 0 cars
                             "HH_cars_1",  # N of HH with 1 car
                             "HH_cars_2",  # N of HH with 2 cars
                             "HH_cars_3+",  # N of HH with 3+ cars
                             "HHT_rent_free",  # N of HH living rent-free
                             "HHT_owned_outright",  # N of HH owning outright
                             "HHT_owned_mortgage",  # N of HH owning with mortgage
                             "HHT_rented_other",  # N of HH renting from other private landlords
                             "HHT_rented_private",  # N of HH renting from private landlords
                             "HHT_shared_ownership",  # N of HH in shared ownership
                             "HHT_rented_social",  # N of HH renting from social landlords
                             "Acc_detached",  # N of HH living in detached houses
                             "Acc_caravan",  # N of HH living in caravans
                             "Acc_commercial",  # N of HH living in commercial buildings
                             "Acc_flat",  # N of HH living in flats
                             "Acc_converted_or_shared",  # N of HH living in converted or shared houses
                             "Acc_converted_other",  # N of HH living in other converted buildings
                             "Acc_semidetached",  # N of HH living in semi-detached houses
                             "Acc_terraced"]  # N of HH living in terraced houses

    demand_summary_table_24 = OLS_analysis(analysis_21_24, dependent_variable, independent_variables)

    # save the summary table
    demand_summary_table_24.to_csv(outputs["OLS_demand_2024"], index=False)

    # __________________________________________________________________________________________________________________
    # OLS for ACCESSIBILITY 2024
    dependent_variable = "accessibility_24"
    independent_variables = ["Med_HP_2023",  # Median house prices 2023 (March)
                             "ASG_AB",  # Approx social grade (higher and intermediate occ.)
                             "ASG_C1",  # Approx social grade (Supervisory and junior managerial  occ.)
                             "ASG_C2",  # Approx social grade (Skilled manual occ.)
                             "ASG_DE",  # Approx social grade (Semi-skilled, unempl., lowest grade occ.)
                             "D0",  # Deprivation index 0 (no dimensions)
                             "D1",  # Deprivation index 1 (1 dimension)
                             "D2",  # Deprivation index 2 (2 dimensions)
                             "D3",  # Deprivation index 3 (3 dimensions)
                             "D4",  # Deprivation index 4 (4 dimensions)
                             "Pop_density",  # Population density
                             "HH_cars_0",  # N of HH with 0 cars
                             "HH_cars_1",  # N of HH with 1 car
                             "HH_cars_2",  # N of HH with 2 cars
                             "HH_cars_3+",  # N of HH with 3+ cars
                             "HHT_rent_free",  # N of HH living rent-free
                             "HHT_owned_outright",  # N of HH owning outright
                             "HHT_owned_mortgage",  # N of HH owning with mortgage
                             "HHT_rented_other",  # N of HH renting from other private landlords
                             "HHT_rented_private",  # N of HH renting from private landlords
                             "HHT_shared_ownership",  # N of HH in shared ownership
                             "HHT_rented_social",  # N of HH renting from social landlords
                             "Acc_detached",  # N of HH living in detached houses
                             "Acc_caravan",  # N of HH living in caravans
                             "Acc_commercial",  # N of HH living in commercial buildings
                             "Acc_flat",  # N of HH living in flats
                             "Acc_converted_or_shared",  # N of HH living in converted or shared houses
                             "Acc_converted_other",  # N of HH living in other converted buildings
                             "Acc_semidetached",  # N of HH living in semi-detached houses
                             "Acc_terraced"]  # N of HH living in terraced houses

    accessibility_summary_table_24 = OLS_analysis(analysis_21_24, dependent_variable, independent_variables)

    # save the summary table
    accessibility_summary_table_24.to_csv(outputs["OLS_accessibility_2024"], index=False)

    # __________________________________________________________________________________________________________________
    # OLS for ACCESSIBILITY DIFFERENCE 2024-2021
    dependent_variable = "acc_diff_24_21"
    independent_variables = ["Med_HP_2021",  # Median house prices 2021 (December)
                             "Med_HP_2023",  # Median house prices 2023 (March)
                             "ASG_AB",  # Approx social grade (higher and intermediate occ.)
                             "ASG_C1",  # Approx social grade (Supervisory and junior managerial  occ.)
                             "ASG_C2",  # Approx social grade (Skilled manual occ.)
                             "ASG_DE",  # Approx social grade (Semi-skilled, unempl., lowest grade occ.)
                             "D0",  # Deprivation index 0 (no dimensions)
                             "D1",  # Deprivation index 1 (1 dimension)
                             "D2",  # Deprivation index 2 (2 dimensions)
                             "D3",  # Deprivation index 3 (3 dimensions)
                             "D4",  # Deprivation index 4 (4 dimensions)
                             "Pop_density",  # Population density
                             "HH_cars_0",  # N of HH with 0 cars
                             "HH_cars_1",  # N of HH with 1 car
                             "HH_cars_2",  # N of HH with 2 cars
                             "HH_cars_3+",  # N of HH with 3+ cars
                             "HHT_rent_free",  # N of HH living rent-free
                             "HHT_owned_outright",  # N of HH owning outright
                             "HHT_owned_mortgage",  # N of HH owning with mortgage
                             "HHT_rented_other",  # N of HH renting from other private landlords
                             "HHT_rented_private",  # N of HH renting from private landlords
                             "HHT_shared_ownership",  # N of HH in shared ownership
                             "HHT_rented_social",  # N of HH renting from social landlords
                             "Acc_detached",  # N of HH living in detached houses
                             "Acc_caravan",  # N of HH living in caravans
                             "Acc_commercial",  # N of HH living in commercial buildings
                             "Acc_flat",  # N of HH living in flats
                             "Acc_converted_or_shared",  # N of HH living in converted or shared houses
                             "Acc_converted_other",  # N of HH living in other converted buildings
                             "Acc_semidetached",  # N of HH living in semi-detached houses
                             "Acc_terraced"]  # N of HH living in terraced houses

    acc_diff_summary_table_24_21 = OLS_analysis(analysis_21_24, dependent_variable, independent_variables)

    # save the summary table
    acc_diff_summary_table_24_21.to_csv(outputs["OLS_diff_accessibility_21_24"], index=False)

# Geographically Weighted Regression (GWR) analysis 2021
'''
GWR_flag_METHOD1 = False
if GWR_flag_METHOD1 == True:
    # import the London centroids shapefile with geopandas
    London_LSOA_centroids = gpd.read_file(inputs["London_LSOA_centroids"])
    # Extract the coordinates of the centroids
    London_LSOA_centroids["x"] = London_LSOA_centroids.centroid.x
    London_LSOA_centroids["y"] = London_LSOA_centroids.centroid.y

    # Only keep the relevant columns
    London_LSOA_centroids = London_LSOA_centroids[["LSOA21CD", "x", "y"]]

    # Merge the centroids with the analysis_21_24 dataframe
    analysis_21_24 = analysis_21_24.merge(London_LSOA_centroids, on="LSOA21CD", how="outer")
    #analysis_21_24.drop(columns=["LSOA21CD"], inplace=True)

    # Remove NaNs
    analysis_21_24 = analysis_21_24.dropna()

    # Step 1: Prepare spatial coordinates and data
    u = analysis_21_24["x"]
    v = analysis_21_24["y"]
    coords = list(zip(u,v))  # Spatial coordinates
    X = analysis_21_24[["Med_HP_2021"]].values  # Independent variables
    Y = analysis_21_24["accessibility"].values.reshape(-1, 1)  # Dependent variable reshaped for GWR

    # Step 2: Select the optimal bandwidth
    selector = Sel_BW(coords, Y, X)
    bandwidth = selector.search()
    print(f"Optimal Bandwidth: {bandwidth}")

    # Step 3: Fit the GWR model
    gwr_results = GWR(coords, Y, X, bandwidth).fit()

    # Step 4: Inspect results
    print(gwr_results.summary())

    # Step 5: Extract local coefficients
    local_coefficients = gwr_results.params
    print("Local Coefficients:\n", local_coefficients)


    London_LSOA_polygons = gpd.read_file(inputs["London_LSOA_polygons"])
    # Now use the polygons geometry for the analysis_21_24 dataframe
    analysis_21_24 = analysis_21_24.merge(London_LSOA_polygons, on="LSOA21CD", how="outer")
    # Now turn the analysis_21_24 dataframe into a geodataframe
    analysis_21_24 = gpd.GeoDataFrame(analysis_21_24)

    # Drop NaNs
    analysis_21_24 = analysis_21_24.dropna()

    # Link to GWR tutorial (follow for visualisation)
    # https://deepnote.com/app/carlos-mendez/PYTHON-GWR-and-MGWR-71dd8ba9-a3ea-4d28-9b20-41cc8a282b7a

    analysis_21_24['gwr_R2'] = gwr_results.localR2

    # Step 6: Visualize local R^2
    
    fig, ax = plt.subplots(figsize=(6, 6))
    analysis_21_24.plot(column='gwr_R2', cmap='coolwarm', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
             legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=ax)
             #legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=ax)
    ax.set_title('Local R2', fontsize=12)
    ax.axis("off")
    # plt.savefig('myMap.png',dpi=150, bbox_inches='tight')
    plt.show()
    

    # Step 7: Visualize local coefficients
    # Add coefficients to the dataframe
    analysis_21_24['gwr_intercept'] = gwr_results.params[:, 0]
    analysis_21_24['gwr_Med_HP_2021'] = gwr_results.params[:, 1]

    # Filter/correct t-stats
    analysis_21_24_filtered_t = gwr_results.filter_tvals(alpha=0.05)
    analysis_21_24_filtered_t_df = pd.DataFrame(aanalysis_21_24_filtered_t)

    # Filter t-values: corrected alpha due to multiple testing
    analysis_21_24_filtered_tc = gwr_results.filter_tvals()
    analysis_21_24_filtered_tc_df = pd.DataFrame(analysis_21_24_filtered_tc)

    # Map coefficients
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    analysis_21_24.plot(column='gwr_Med_HP_2021', cmap='coolwarm', linewidth=0.01, scheme='FisherJenks', k=5, legend=True,
             legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[0])

    analysis_21_24.plot(column='gwr_Med_HP_2021', cmap='coolwarm', linewidth=0.05, scheme='FisherJenks', k=5, legend=False,
             legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[1])
    analysis_21_24[analysis_21_24_filtered_t_df[:, 1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[1])

    analysis_21_24.plot(column='gwr_Med_HP_2021', cmap='coolwarm', linewidth=0.05, scheme='FisherJenks', k=5, legend=False,
             legend_kwds={'bbox_to_anchor': (1.10, 0.96)}, ax=axes[2])
    analysis_21_24[analysis_21_241_filtered_tc_df[:, 1] == 0].plot(color='white', linewidth=0.05, edgecolor='black', ax=axes[2])

    plt.tight_layout()

    axes[0].axis("off")
    axes[1].axis("off")
    axes[2].axis("off")

    axes[0].set_title('(a) GWR: Med_HP_2021 (BW: ' + str(bandwidth) + '), all coeffs', fontsize=12)
    axes[1].set_title('(b) GWR: Med_HP_2021 (BW: ' + str(bandwidth) + '), significant coeffs', fontsize=12)
    axes[2].set_title('(c) GWR: Med_HP_2021 (BW: ' + str(bandwidth) + '), significant coeffs and corr. p-values',
                      fontsize=12)
    plt.show()
'''
# GWR analysis 2021 (METHOD 2)
# See here: https://github.com/urschrei/Geopython/blob/master/geographically_weighted_regression.ipynb

