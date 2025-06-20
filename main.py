"""
Measuring supply and demand of electric vehicle charging infrastructure (EVCI) in London

Author Fulvio D. Lopane
Centre for Advanced Spatial Analysis

started coding: October 2024
"""

import pandas as pd
import geopandas as gpd
from duckdb import values

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
from spglm.family import Gaussian
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
from matplotlib.colors import LinearSegmentedColormap



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
London_LSOAs = pd.read_csv(inputs["London_LSOAs"])
London_LSOA_codes = London_LSOAs["LSOA21CD"].tolist()

# Filter out London from GB mean house prices
if not os.path.exists(generated["Mean_house_prices_London"]):
    HP_GB = pd.read_csv(inputs["Mean_house_prices_GB"])
    HP_London = HP_GB[HP_GB["LSOA code"].isin(London_LSOA_codes)]
    HP_London.rename(columns={"LSOA code": "LSOA21CD"}, inplace=True)
    HP_London.to_csv(generated["Mean_house_prices_London"], index=False)

# Filter out London from GB median house prices
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
# Remove commas "," in the field to turn the Population density to float
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
                                         "Social rented: Other social rented": "HHT_rented_social_other",
                                         "Lives rent free": "HHT_rent_free",
                                         "Social rented: Rents from council or Local Authority": "HHT_rented_social"}, inplace=True)

House_tenure_London_2021 = House_tenure_London_2021[["LSOA21CD", "HHT_rent_free", "HHT_owned_outright",
                                                     "HHT_owned_mortgage", "HHT_rented_other", "HHT_rented_private",
                                                     "HHT_shared_ownership", "HHT_rented_social_other", "HHT_rented_social"]]

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
analysis_21_24["acc_diff_24_21"] = analysis_21_24["accessibility_24"] - analysis_21_24["accessibility_21"]

# replace zeroes with very small numbers in EVCI2021 and y2021Q4 to avoid division by zero
analysis_21_24["EVCI2021"] = analysis_21_24["EVCI2021"].replace(0, 0.0000001)
analysis_21_24["y2021Q4"] = analysis_21_24["y2021Q4"].replace(0, 0.0000001)

# Create demand and supply improvements between 2021 and 2024
analysis_21_24["EVCI_improvement"] = analysis_21_24["EVCI2024"] - analysis_21_24["EVCI2021"]
analysis_21_24["EVCI_improvement_rate"] = analysis_21_24["EVCI_improvement"] / analysis_21_24["EVCI2021"]
analysis_21_24["EVCI_improvement_rate"] = analysis_21_24["EVCI_improvement_rate"].replace([np.inf, -np.inf], np.nan)
#analysis_21_24["EVCI_improvement_rate"] = analysis_21_24["EVCI_improvement_rate"].fillna(0)

analysis_21_24["EVCI_improvement_arcsinh"] = np.arcsinh(analysis_21_24["EVCI2024"]) - np.arcsinh(analysis_21_24["EVCI2021"])
analysis_21_24["EVCI_improvement_rate_arcsinh"] = analysis_21_24["EVCI_improvement_arcsinh"] / np.arcsinh(analysis_21_24["EVCI2021"])

analysis_21_24["EV_licensing_improvement"] = analysis_21_24["y2024Q2"] - analysis_21_24["y2021Q4"]
analysis_21_24["EV_licensing_improvement_rate"] = analysis_21_24["EV_licensing_improvement"] / analysis_21_24["y2021Q4"]
analysis_21_24["EV_licensing_improvement_rate"] = analysis_21_24["EV_licensing_improvement_rate"].replace([np.inf, -np.inf], np.nan)
#analysis_21_24["EV_licensing_improvement_rate"] = analysis_21_24["EV_licensing_improvement_rate"].fillna(0)

analysis_21_24["EV_licensing_improvement_arcsinh"] = np.arcsinh(analysis_21_24["y2024Q2"]) - np.arcsinh(analysis_21_24["y2021Q4"])
analysis_21_24["EV_licensing_improvement_rate_arcsinh"] = analysis_21_24["EV_licensing_improvement_arcsinh"] / np.arcsinh(analysis_21_24["y2021Q4"])

''' ### TESTS FOR NORMALITY OF THE VARIABLES
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm
# calculate mean and standard deviation of EV_licensing_improvement_rate
# remove NaN values
analysis_21_24 = analysis_21_24.dropna(subset=["EV_licensing_improvement"])

mean_EV_licensing_improvement_rate = analysis_21_24["EV_licensing_improvement"].mean()
std_EV_licensing_improvement_rate = analysis_21_24["EV_licensing_improvement"].std()

print(mean_EV_licensing_improvement_rate, std_EV_licensing_improvement_rate)

#analysis_21_24["EV_licensing_improvement_NORM"] = (analysis_21_24["EV_licensing_improvement"]-mean_EV_licensing_improvement_rate)/std_EV_licensing_improvement_rate

scaler= MinMaxScaler()

analysis_21_24["EV_licensing_improvement_NORM"] = scaler.fit_transform(analysis_21_24["EV_licensing_improvement"].values.reshape(-1,1))
print(analysis_21_24["EV_licensing_improvement_NORM"])
# count homw many nan values are in the column
print(analysis_21_24["EV_licensing_improvement_NORM"].isnull().sum())

mean_EV_licensing_improvement_NORM_rate = analysis_21_24["EV_licensing_improvement_NORM"].mean()
std_EV_licensing_improvement_NORM_rate = analysis_21_24["EV_licensing_improvement_NORM"].std()
# print the mean and standard deviation of the normalized variable
print("NORM variable" , mean_EV_licensing_improvement_NORM_rate, std_EV_licensing_improvement_NORM_rate)

# remove NaN values
analysis_21_24 = analysis_21_24.dropna(subset=["EV_licensing_improvement_NORM"])

hist, bins = np.histogram(analysis_21_24["EV_licensing_improvement_NORM"], bins=200, density= True)
x = (bins[1:] + bins[:-1]) / 2
x_gauss = np.linspace(-5, 5, 100)
pdf_gauss = norm.pdf(x_gauss, loc=0, scale=1)
plt.semilogy(x, hist)
plt.semilogy(x_gauss, pdf_gauss, 'r')
plt.show()


variables_to_test = analysis_21_24[["EV_licensing_improvement_NORM"]]

# Check normality of the dependent variables
check_normality_flag = True
if check_normality_flag == True:
    from scipy import stats
    for var in variables_to_test:
        # Shapiro-Wilk test for normality
        stat, p = stats.shapiro(analysis_21_24[var])
        print("--------------------------------------------")
        print(f"Shapiro-Wilk test for normality for {var}:")
        print(f"Statistics={stat:.3f}, p={p:.3f}")
        # Interpret
        alpha = 0.05
        if p > alpha:
            print(f"{var} sample looks Gaussian (fail to reject H0)")
        else:
            print(f"{var} sample does not look Gaussian (reject H0)")
'''

# Remove columns
#analysis_21_24.drop(columns=["LSOA21CD", "LSOA21NM", "HH_number", "EVCI2021", "y2021Q4"], inplace=True)
#analysis_21_24.drop(columns=["LSOA21CD", "LSOA21NM", "HH_number"], inplace=True)
#analysis_21_24.drop(columns=["LSOA21NM", "HH_number"], inplace=True)

# Reduce the number of variables by collapsing some of them
'''
- Approx social grade (ASG): combine the 4 categories into 2: AB and CDE
- Deprivation index: only keep 3+ dimensions (in a single variable)
- Vehicle ownership: combine the 4 categories into 2: 0-1 cars and 2+ cars
- House tenure: combine the 8 categories into 3: owned, rented, and other
- Accommodation type: combine the 8 categories into 3: detached+semidetached, flat, and other
'''

# Approx social grade (ASG)
analysis_21_24["ASG_AB_C1"] = analysis_21_24["ASG_AB"] + analysis_21_24["ASG_C1"]
analysis_21_24["ASG_C2_DE"] = analysis_21_24["ASG_C2"] + analysis_21_24["ASG_DE"]

# Deprivation index
#analysis_21_24["D2+"] = analysis_21_24["D2"] + analysis_21_24["D3"] + analysis_21_24["D4"]
analysis_21_24["D3+"] = analysis_21_24["D3"] + analysis_21_24["D4"]

# Vehicle ownership
analysis_21_24["HH_cars_0_1"] = analysis_21_24["HH_cars_0"] + analysis_21_24["HH_cars_1"]
analysis_21_24["HH_cars_2+"] = analysis_21_24["HH_cars_2"] + analysis_21_24["HH_cars_3+"]

# House tenure
analysis_21_24["HHT_owned"] = (analysis_21_24["HHT_owned_outright"]
                               + analysis_21_24["HHT_owned_mortgage"]
                               + analysis_21_24["HHT_shared_ownership"])

analysis_21_24["HHT_rented"] = (analysis_21_24["HHT_rented_other"]
                                + analysis_21_24["HHT_rented_private"]
                                + analysis_21_24["HHT_rented_social_other"]
                                + analysis_21_24["HHT_rented_social"])

# Accommodation type
analysis_21_24["Acc_detached_semidet"] = (analysis_21_24["Acc_detached"]
                                            + analysis_21_24["Acc_semidetached"])

analysis_21_24["Acc_flat"] = (analysis_21_24["Acc_flat"]
                              + analysis_21_24["Acc_converted_or_shared"]
                              + analysis_21_24["Acc_converted_other"])

analysis_21_24["Acc_other"] = (analysis_21_24["Acc_caravan"]
                               + analysis_21_24["Acc_commercial"]
                               + analysis_21_24["Acc_terraced"])
'''
# change the units of my variables to make the coefficients bigger
# but first check the magnitude of the variables with some print statements
print("Median house prices 2021: ", analysis_21_24["Med_HP_2021"].min(), analysis_21_24["Med_HP_2021"].max())
print("Median house prices 2023: ", analysis_21_24["Med_HP_2023"].min(), analysis_21_24["Med_HP_2023"].max())
print("Approx social grade AB+C1: ", analysis_21_24["ASG_AB_C1"].min(), analysis_21_24["ASG_AB_C1"].max())
print("Approx social grade C2+DE: ", analysis_21_24["ASG_C2_DE"].min(), analysis_21_24["ASG_C2_DE"].max())
print("Deprivation index 2+: ", analysis_21_24["D2+"].min(), analysis_21_24["D2+"].max())
print("Population density: ", analysis_21_24["Pop_density"].min(), analysis_21_24["Pop_density"].max())
print("HH cars 0-1: ", analysis_21_24["HH_cars_0_1"].min(), analysis_21_24["HH_cars_0_1"].max())
print("HH cars 2+: ", analysis_21_24["HH_cars_2+"].min(), analysis_21_24["HH_cars_2+"].max())
print("HHT owned: ", analysis_21_24["HHT_owned"].min(), analysis_21_24["HHT_owned"].max())
print("HHT rented: ", analysis_21_24["HHT_rented"].min(), analysis_21_24["HHT_rented"].max())
print("Acc detached+semidet: ", analysis_21_24["Acc_detached_semidet"].min(), analysis_21_24["Acc_detached_semidet"].max())
print("Acc flat: ", analysis_21_24["Acc_flat"].min(), analysis_21_24["Acc_flat"].max())
print("Acc other: ", analysis_21_24["Acc_other"].min(), analysis_21_24["Acc_other"].max())

# print the min and max also for the dependent variables
print("EVCI 2021: ", analysis_21_24["EVCI2021"].min(), analysis_21_24["EVCI2021"].max())
print("EVCI 2024: ", analysis_21_24["EVCI2024"].min(), analysis_21_24["EVCI2024"].max())
print("EV licensing 2021: ", analysis_21_24["y2021Q4"].min(), analysis_21_24["y2021Q4"].max())
print("EV licensing 2024: ", analysis_21_24["y2024Q2"].min(), analysis_21_24["y2024Q2"].max())
print("Accessibility 2021: ", analysis_21_24["accessibility_21"].min(), analysis_21_24["accessibility_21"].max())
print("Accessibility 2024: ", analysis_21_24["accessibility_24"].min(), analysis_21_24["accessibility_24"].max())
print("Accessibility difference 2024-2021: ", analysis_21_24["acc_diff_24_21"].min(), analysis_21_24["acc_diff_24_21"].max())
'''

thousands_flag = False
if thousands_flag == True:
    analysis_21_24["ASG_AB_C1"] = analysis_21_24["ASG_AB_C1"] / 1000
    analysis_21_24["ASG_C2_DE"] = analysis_21_24["ASG_C2_DE"] / 1000
    analysis_21_24["D3+"] = analysis_21_24["D2+"] / 1000
    analysis_21_24["Pop_density"] = analysis_21_24["Pop_density"] / 1000
    analysis_21_24["HH_cars_0_1"] = analysis_21_24["HH_cars_0_1"] / 1000
    analysis_21_24["HH_cars_2+"] = analysis_21_24["HH_cars_2+"] / 1000
    analysis_21_24["HHT_owned"] = analysis_21_24["HHT_owned"] / 1000
    analysis_21_24["HHT_rented"] = analysis_21_24["HHT_rented"] / 1000
    analysis_21_24["Acc_detached_semidet"] = analysis_21_24["Acc_detached_semidet"] / 1000
    analysis_21_24["Acc_flat"] = analysis_21_24["Acc_flat"] / 1000
    analysis_21_24["Acc_other"] = analysis_21_24["Acc_other"] / 1000

# Create variables containing shares instead of absolute numbers
var_shares_flag = True
if var_shares_flag == True:
    analysis_21_24["s-ASG_ABC1"] = analysis_21_24["ASG_AB_C1"] / (analysis_21_24["ASG_AB_C1"] + analysis_21_24["ASG_C2_DE"])
    analysis_21_24["s-ASG_C2DE"] = analysis_21_24["ASG_C2_DE"] / (analysis_21_24["ASG_AB_C1"] + analysis_21_24["ASG_C2_DE"])
    analysis_21_24["s-D3+"] = analysis_21_24["D3+"] / (analysis_21_24["D3+"] + analysis_21_24["D0"] + analysis_21_24["D1"] + analysis_21_24["D2"])
    analysis_21_24["s-HHcars_01"] = analysis_21_24["HH_cars_0_1"] / (analysis_21_24["HH_cars_0_1"] + analysis_21_24["HH_cars_2+"])
    analysis_21_24["s-HHcars_2+"] = analysis_21_24["HH_cars_2+"] / (analysis_21_24["HH_cars_0_1"] + analysis_21_24["HH_cars_2+"])
    analysis_21_24["s-HHT_owned"] = analysis_21_24["HHT_owned"] / (analysis_21_24["HHT_owned"] + analysis_21_24["HHT_rented"])
    analysis_21_24["s-HHT_rented"] = analysis_21_24["HHT_rented"] / (analysis_21_24["HHT_owned"] + analysis_21_24["HHT_rented"])
    analysis_21_24["s-Acc_det-semidet"] = analysis_21_24["Acc_detached_semidet"] / (analysis_21_24["Acc_detached_semidet"] + analysis_21_24["Acc_flat"] + analysis_21_24["Acc_other"])
    analysis_21_24["s-Acc_flat"] = analysis_21_24["Acc_flat"] / (analysis_21_24["Acc_detached_semidet"] + analysis_21_24["Acc_flat"] + analysis_21_24["Acc_other"])
    analysis_21_24["s-Acc_other"] = analysis_21_24["Acc_other"] / (analysis_21_24["Acc_detached_semidet"] + analysis_21_24["Acc_flat"] + analysis_21_24["Acc_other"])

# Add employment distribution information
Employment_2021_GB = pd.read_csv(inputs["Employment_2021_GB"])
Employment_2021_London = Employment_2021_GB[Employment_2021_GB["LSOACD"].isin(London_LSOA_codes)]
Employment_2021_London = Employment_2021_London[["LSOACD", "total"]]
Employment_2021_London.rename(columns={"LSOACD": "LSOA21CD", "total": "jobs_2021"}, inplace=True)

Employment_2023_GB = pd.read_csv(inputs["Employment_2023_GB"])
Employment_2023_London = Employment_2023_GB[Employment_2023_GB["LSOACD"].isin(London_LSOA_codes)]
Employment_2023_London = Employment_2023_London[["LSOACD", "total"]]
Employment_2023_London.rename(columns={"LSOACD": "LSOA21CD", "total": "jobs_2023"}, inplace=True)


# Transform the jobs in thousands of jobs
job_thousands_flag = True
if job_thousands_flag == True:
    Employment_2021_London["job_th_21"] = Employment_2021_London["jobs_2021"] / 1000
    Employment_2023_London["job_th_23"] = Employment_2023_London["jobs_2023"] / 1000

# Merge the employment data
analysis_21_24 = analysis_21_24.merge(Employment_2021_London, on="LSOA21CD", how="outer")
analysis_21_24 = analysis_21_24.merge(Employment_2023_London, on="LSOA21CD", how="outer")

# LSOA boundaries polygons
lsoa_gdf = gpd.read_file(inputs["London_LSOA_polygons"])

# Calculate road network characteristics in London and add it as an extra variable
# df columns: 'street_length_total', 'street_density_km', 'intersection_count', 'intersection_density_km'
if not os.path.exists(generated["road_net_chars"]):
    # use the function defined in utils: road_net_chars_calculator
    road_chars = road_net_chars_calculator(lsoa_gdf) # VERY time-consuming: 8 hours on a laptop
    # save the road network characteristics to a csv file
    road_chars.to_csv(generated["road_net_chars"], index=False)
else:
    road_chars = pd.read_csv(generated["road_net_chars"])

# ----------------------------------------------------------------------------------------------------------------------
# MANUAL FIX of "error" LSOAs to avoid another run of the road network characteristics calculation (8 hours run)
# LSOAs to fix: ['E01000826', 'E01003629', 'E01003730', 'E01004054', 'E01033917']

# Manually enter the correct vales for each column of the LSOAs
road_chars.loc[road_chars["LSOA21CD"] == "E01000826", "street_length_total"] = 7.524
road_chars.loc[road_chars["LSOA21CD"] == "E01000826", "street_density_km"] = 7.524/(513343/1000000)
road_chars.loc[road_chars["LSOA21CD"] == "E01000826", "intersection_count"] = 22.0
road_chars.loc[road_chars["LSOA21CD"] == "E01000826", "intersection_density_km"] = 22.0/(513343/1000000)

road_chars.loc[road_chars["LSOA21CD"] == "E01003629", "street_length_total"] = 1.241
road_chars.loc[road_chars["LSOA21CD"] == "E01003629", "street_density_km"] = 1.241/(64471.4/1000000)
road_chars.loc[road_chars["LSOA21CD"] == "E01003629", "intersection_count"] = 0.0
road_chars.loc[road_chars["LSOA21CD"] == "E01003629", "intersection_density_km"] = 0.0

road_chars.loc[road_chars["LSOA21CD"] == "E01003730", "street_length_total"] = 2.0
road_chars.loc[road_chars["LSOA21CD"] == "E01003730", "street_density_km"] = 2.0/(89400.4/1000000)
road_chars.loc[road_chars["LSOA21CD"] == "E01003730", "intersection_count"] = 2.0
road_chars.loc[road_chars["LSOA21CD"] == "E01003730", "intersection_density_km"] = 2.0/(89400.4/1000000)

road_chars.loc[road_chars["LSOA21CD"] == "E01004054", "street_length_total"] = 2.575
road_chars.loc[road_chars["LSOA21CD"] == "E01004054", "street_density_km"] = 2.575/(116254.4/1000000)
road_chars.loc[road_chars["LSOA21CD"] == "E01004054", "intersection_count"] = 22.0
road_chars.loc[road_chars["LSOA21CD"] == "E01004054", "intersection_density_km"] = 22.0/(116254.4/1000000)

road_chars.loc[road_chars["LSOA21CD"] == "E01033917", "street_length_total"] = 0.582
road_chars.loc[road_chars["LSOA21CD"] == "E01033917", "street_density_km"] = 0.582/(32612.5/1000000)
road_chars.loc[road_chars["LSOA21CD"] == "E01033917", "intersection_count"] = 3.0
road_chars.loc[road_chars["LSOA21CD"] == "E01033917", "intersection_density_km"] = 3.0/(32612.5/1000000)
# ----------------------------------------------------------------------------------------------------------------------

# if any of the rows of road_chars contains "error", then print the value of the corresponding LSOA code
error_flag = True
error_list = []
if error_flag == True:
    for i in range(len(road_chars)):
        if "error" in road_chars.iloc[i].values:
            # append the LSOA code to the error list
            error_list.append(road_chars.iloc[i]["LSOA21CD"])
            #print("WARNING - There is an error in LSOA: ", road_chars.iloc[i])
            #print()
if len(error_list) > 0:
    print("LSOAs with errors in road chars: ", error_list)
    print()

# convert the values of the columns to float
road_chars["street_length_total"] = road_chars["street_length_total"].astype(float)
road_chars["street_density_km"] = road_chars["street_density_km"].astype(float)
road_chars["intersection_count"] = road_chars["intersection_count"].astype(float)
road_chars["intersection_density_km"] = road_chars["intersection_density_km"].astype(float)

# Only keep the 'street_density_km'column
road_chars = road_chars[["LSOA21CD", "street_density_km"]]

# rename columns
road_chars.rename(columns={"street_density_km": "RoadKmDen"}, inplace=True)

# Merge the road network characteristics to the analysis dataframe
analysis_21_24 = analysis_21_24.merge(road_chars, on="LSOA21CD", how="outer")

# Calculate amenities (POIs) density per LSOA in London and add it as an extra variable
if not os.path.exists(generated["POIs"]):
    # use the function defined in utils: amenities_scarping
    POI_df = amenities_scarping(lsoa_gdf)
    # save the road network characteristics to a csv file
    POI_df.to_csv(generated["POIs"], index=False)
else:
    POI_df = pd.read_csv(generated["POIs"])

# Generate a df with the POI count per LSOA
POI_lsoa = POI_df.groupby("LSOA21CD").size().reset_index(name="POI_count")

# Using the lsoa_gdf geomtery, calculate the area of each LSOA in sq km
lsoa_gdf["area_sqkm"] = lsoa_gdf["geometry"].to_crs(epsg=27700).area / 10**6
# now calculate the density of POIs per sq km
POI_lsoa = POI_lsoa.merge(lsoa_gdf[["LSOA21CD", "area_sqkm"]], on="LSOA21CD", how="outer")
POI_lsoa["POI_dens"] = POI_lsoa["POI_count"] / POI_lsoa["area_sqkm"] # POIs per sq km

# Merge the POI density to the analysis dataframe
analysis_21_24 = analysis_21_24.merge(POI_lsoa[["LSOA21CD", "POI_dens"]], on="LSOA21CD", how="outer")

# check null values in the analysis dataframe
#print()
#print("###############################################################################################################")
#print("Number of missing values in the analysis dataframe:")
#print(analysis_21_24.isnull().sum())
#print("###############################################################################################################")
#print()

# check non-numeric values in the analysis dataframe
#print("###############################################################################################################")
#print("Non-numeric values in the analysis dataframe:")
#print(analysis_21_24.select_dtypes(exclude=[np.number]).columns)
#print("###############################################################################################################")
#print()

# Create a df with only the variables for the OLS and GWR analysis
Regression_21_24 = analysis_21_24[["LSOA21CD",                              # LSOA code
                                   "LSOA21NM",                              # LSOA name
                                   "accessibility_21",                      # accessibility 2021
                                   "accessibility_24",                      # accessibility 2024
                                   "acc_diff_24_21",                        # dep var (accessibility)
                                   "EVCI_improvement",                      # supply improvement
                                   "EVCI_improvement_rate",                 # supply improvement rate
                                   #"EVCI_improvement_rate_arcsinh",         # supply improvement rate arcsinh
                                   "EV_licensing_improvement",              # demand improvement
                                   "EV_licensing_improvement_rate",         # demand improvement rate
                                   #"EV_licensing_improvement_rate_arcsinh", # demand improvement rate arcsinh
                                   "s-ASG_ABC1",                            # ASG AB C1
                                   "s-ASG_C2DE",                            # ASG C2 DE
                                   "s-D3+",                                 # HH deprived in 3+ dimensions
                                   "s-HHcars_01",                           # Vehicle ownership (HH w/ 0 or 1 car)
                                   "s-HHcars_2+",                           # Vehicle ownership (HH w/ 2+ cars)
                                   "s-HHT_owned",                           # House tenure: owned
                                   "s-HHT_rented",                          # House tenure: rented
                                   "s-Acc_det-semidet",                     # Accommodation type: detached and semidet.
                                   "s-Acc_flat",                            # Accommodation type: flat
                                   "s-Acc_other",                           # Accommodation type: terraced and other
                                   "Med_HP_2021",                           # Median house prices 2021
                                   "Med_HP_2023",                           # Median house prices 2023
                                   "Pop_density",                           # Population density
                                   "POI_dens"]]                             # POI density
                                   # don't consider the following variables as not statistically significant
                                   #"RoadKmDen",                             # Road network (km) density
                                   #"job_th_21",                             # Thousands of jobs per LSOA 2021
                                   #"job_th_23"]]                            # Thousands of jobs per LSOA 2023

# rename columns longer than 10 characters
Regression_21_24.rename(columns={"accessibility_21": "acc_21",
                                 "accessibility_24": "acc_24",
                                 "acc_diff_24_21": "acc_diff",
                                 "EVCI_improvement": "s_impr",
                                 "EVCI_improvement_rate": "s_imp_rate",
                                 "EV_licensing_improvement": "d_impr",
                                 "EV_licensing_improvement_rate": "d_imp_rate",
                                 "s-HHcars_01": "s-HHcars01",
                                 "s-HHcars_2+": "s-HHcars2+",
                                 "s-HHT_owned": "s-HHT_own",
                                 "s-HHT_rented": "s-HHT_rent",
                                 "s-Acc_det-semidet": "s-semi-det",
                                 "s-Acc_flat": "s-flat",
                                 "s-Acc_other": "s-terr-oth",
                                 "Med_HP_2021": "Med_HP_21",
                                 "Med_HP_2023": "Med_HP_23",
                                 "Pop_density": "Pop_dens"}, inplace=True)

# OLS analysis
OLS_2021_2024_flag = False
# Normalisation options:
#normalise_dependent_variables = False
#normalise_independent_variables = False

if OLS_2021_2024_flag == True:
    # print if the dependent and independent variables are normalised according to the options
    #print("-------------------------------------------------------------------")
    #print("Normalise dependent variables: ", normalise_dependent_variables)
    #print("Normalise independent variables: ", normalise_independent_variables)
    #print("-------------------------------------------------------------------")

    # take the Log of house prices (and other big numbers to avoid coefficients with many zeros)
    print()
    print("Log-transforming the median house prices...")
    print()
    Regression_21_24["Med_HP_21"] = np.log(Regression_21_24["Med_HP_21"])
    Regression_21_24["Med_HP_23"] = np.log(Regression_21_24["Med_HP_23"])

    London_LSOA_centroids = gpd.read_file(inputs["London_LSOA_centroids"])
    # Now use the polygons geometry for the Regression_21_24 dataframe
    Regression_21_24 = Regression_21_24.merge(London_LSOA_centroids, on="LSOA21CD", how="outer")
    # Now turn the Regression_21_24 dataframe into a geodataframe
    Regression_21_24 = gpd.GeoDataFrame(Regression_21_24)

    # Rename columns
    Regression_21_24.rename(columns={"LSOA21NM_x": "LSOA21NM"}, inplace=True)
    # Drop extra columns
    Regression_21_24.drop(columns=["LSOA21NM_y", "GlobalID"], inplace=True)

    # Remove NaNs
    Regression_21_24 = Regression_21_24.dropna()

    # Turn the geometry column into a x and y column
    Regression_21_24["x"] = Regression_21_24.centroid.x
    Regression_21_24["y"] = Regression_21_24.centroid.y

    # categorise the variables in dependent and independent variables and save the categorisation into two lists:
    cat_dep_variables = ["acc_21",                      # Accessibility 2021
                         "acc_24",                      # Accessibility 2024
                         "acc_diff",                        # Accessibility difference 2024 - 2021
                         "s_impr",                     # EVCI (supply) improvement
                         #"s_imp_rate",                 # EVCI (supply) improvement rate
                         "d_impr"]             # EV licensing (demand) improvement
                         #"d_imp_rate"]         # EV licensing (demand) improvement rate
                         #"EVCI2021",                             # EVCI 2021
                         #"y2021Q4",                              # EV licensing 2021
                         #"EVCI2024",                             # EVCI 2024
                         #"y2024Q2"]                              # EV licensing 2024

    cat_indep_variables = ["s-ASG_ABC1",        # Share of HH in social grade AB and C1
                           "s-ASG_C2DE",       # Share of HH in social grade C2 and DE
                           "s-D3+",             # Share of HH deprived in 3+ dimensions
                           "s-HHcars01",      # Share of HH with 0 or 1 car
                           "s-HHcars2+",       # Share of HH with 2+ cars
                           "s-HHT_own",       # share of HH owning outright + mortgage + shared ownership
                           "s-HHT_rent",     # share of HH renting
                           "s-semi-det", # Share of HH living in detached and semidetached houses
                           "s-flat",        # Share of HH living in flats
                           "s-terr-oth",       # Share of HH living in terraced & other houses
                           "Med_HP_21",      # Median house prices 2021 (December)
                           "Med_HP_23",       # Median house prices 2023 (March)
                           "Pop_dens",       # Population density (thousands)
                           #"RoadKmDen",         # Road network (km) density
                           "POI_dens"]          # POI density
                           #"job_th_21",         # Thousands of jobs per LSOA in 2021
                            #"job_th_23"]         # Thousands of jobs per LSOA in 2023

    # use min max scaling for all the variables
    for i in cat_indep_variables:
        Regression_21_24[i] = (Regression_21_24[i] - Regression_21_24[i].min()) / (
                Regression_21_24[i].max() - Regression_21_24[i].min())

    for i in cat_dep_variables:
        Regression_21_24[i] = (Regression_21_24[i] - Regression_21_24[i].min()) / (
                Regression_21_24[i].max() - Regression_21_24[i].min())

    ''' # if using total counts instead of shares:
                           "Med_HP_2021",             # Median house prices 2021 (December)
                           "Med_HP_2023",             # Median house prices 2023 (March)
                           "ASG_AB_C1",               # Approx social grade (higher and intermediate + Supervisory and junior managerial  occ.)
                           "ASG_C2_DE",               # Approx social grade (Skilled manual + Semi-skilled, unempl., lowest grade occ.)
                           "D2+",                     # Deprivation index: n of HH deprived in 2+ dimensions
                           "Pop_density",             # Population density
                           "HH_cars_0_1",             # N of HH with 0 or 1 car
                           "HH_cars_2+",              # N of HH with 2+ cars
                           "HHT_owned",               # N of HH owning outright + mortgage + shared ownership
                           "HHT_rented",              # N of HH renting
                           "Acc_detached_semidet",    # N of HH living in detached and semidetached houses
                           "Acc_flat",                # N of HH living in flats
                           "Acc_other"]               # N of HH living in terraced & other
                           '''

    '''
    if normalise_dependent_variables == True:
        for i in cat_dep_variables:
            Regression_21_24[i] = (Regression_21_24[i] - Regression_21_24[i].min()) / (
                    Regression_21_24[i].max() - Regression_21_24[i].min())

    if normalise_independent_variables == True:
        for i in cat_indep_variables:
            Regression_21_24[i] = (Regression_21_24[i] - Regression_21_24[i].min()) / (
                    Regression_21_24[i].max() - Regression_21_24[i].min())
    '''

    def normalise_and_run_OLS(var, normalise_dependent_variables, normalise_independent_variables):
        # if dependent variables are not normalised, I have to normalise them here if they are included in the independent variables
        if normalise_dependent_variables == False:
            if normalise_independent_variables == True:
                # copy the analysis dataframe into a new dataframe for the supply_21 analysis
                Regression_21_24_n = Regression_21_24.copy()
                # for each variable in var, normalise it:
                for v in var:
                    Regression_21_24_n[v] = (Regression_21_24_n[v] - Regression_21_24_n[v].min()) / (
                            Regression_21_24_n[v].max() - Regression_21_24_n[v].min())
                # run the OLS analysis with the Regression_21_24 dataframe
                supply_summary_table = OLS_analysis(Regression_21_24_n, dependent_variable, independent_variables)
                return supply_summary_table
            else:
                supply_summary_table = OLS_analysis(Regression_21_24, dependent_variable, independent_variables)
                return supply_summary_table
        else:
            if normalise_independent_variables == True:
                supply_summary_table = OLS_analysis(Regression_21_24, dependent_variable, independent_variables)
                return supply_summary_table
            else:
                # throw an exception as some dependent variables are normalised and they are included in the non-normalised independent variables
                raise Exception("Dependent variables are normalised, but some of them are included in the non-normalised independent variables."
                                "Please change the normalisation options.")

    # __________________________________________________________________________________________________________________
    # OLS for ACCESSIBILITY 2021
    dependent_variable = "acc_21"
    independent_variables = ["s-ASG_ABC1",        # Share of HH in social grade AB and C1
                           "s-ASG_C2DE",       # Share of HH in social grade C2 and DE
                           "s-D3+",             # Share of HH deprived in 3+ dimensions
                           "s-HHcars01",      # Share of HH with 0 or 1 car
                           "s-HHcars2+",       # Share of HH with 2+ cars
                           "s-HHT_own",       # share of HH owning outright + mortgage + shared ownership
                           "s-HHT_rent",     # share of HH renting
                           "s-semi-det", # Share of HH living in detached and semidetached houses
                           "s-flat",        # Share of HH living in flats
                           "s-terr-oth",       # Share of HH living in terraced & other houses
                           "Med_HP_21",      # Median house prices 2021 (December)
                           #"Med_HP_23",       # Median house prices 2023 (March)
                           "Pop_dens",       # Population density (thousands)
                           #"RoadKmDen",         # Road network (km) density
                           "POI_dens"]          # POI density
                           #"job_th_21"]         # Thousands of jobs per LSOA in 2021
                           #"job_th_23"]         # Thousands of jobs per LSOA in 2023

    multiv_independent_variables = [#"s-ASG_ABC1",  # Share of HH in social grade AB and C1
                             "s-ASG_C2DE",  # Share of HH in social grade C2 and DE
                             "s-D3+",  # Share of HH deprived in 3+ dimensions
                             "s-HHcars01",  # Share of HH with 0 or 1 car
                             #"s-HHcars2+",  # Share of HH with 2+ cars
                             #"s-HHT_own",  # share of HH owning outright + mortgage + shared ownership
                             "s-HHT_rent",  # share of HH renting
                             #"s-semi-det",  # Share of HH living in detached and semidetached houses
                             "s-flat",  # Share of HH living in flats
                             "s-terr-oth",  # Share of HH living in terraced & other houses
                             "Med_HP_21",  # Median house prices 2021 (December)
                             # "Med_HP_23",       # Median house prices 2023 (March)
                             "Pop_dens",  # Population density (thousands)
                             # "RoadKmDen",         # Road network (km) density
                             "POI_dens"]  # POI density
                            # "job_th_21"]         # Thousands of jobs per LSOA in 2021
                            # "job_th_23"]         # Thousands of jobs per LSOA in 2023

    # Multivariate regression analysis
    #accessibility_summary_table_21 = OLS_analysis(Regression_21_24, dependent_variable, independent_variables)
    accessibility_summary_table_21, accessibility_21_residuals, accessibility_21_moran = OLS_analysis_multivariate_moran(Regression_21_24, dependent_variable, independent_variables)
    # save the summary table
    accessibility_summary_table_21.to_csv(outputs["OLS_accessibility_2021"], index=False)

    # save the residuals
    # Create a copy of the cleaned data used for modeling
    accessibility_21_residuals_df = Regression_21_24.copy().reset_index(drop=True)
    # Convert residuals to a pandas Series with the same index
    residuals_series = pd.Series(accessibility_21_residuals, name='residuals')
    # Add residuals column to the DataFrame
    accessibility_21_residuals_df['residuals'] = residuals_series
    accessibility_21_residuals_df[["residuals"]].to_csv(outputs["OLS_accessibility_2021_residuals"], index=False)

    # save the moran's I results
    # Create a DataFrame with Moran's I results
    moran_results_accessibility21 = pd.DataFrame({
        'Moran_I': [accessibility_21_moran.I],
        'Expected_I': [accessibility_21_moran.EI],
        'Variance': [accessibility_21_moran.VI_norm],
        'z_score': [accessibility_21_moran.z_norm],
        'p_value': [accessibility_21_moran.p_norm],
        'p_sim': [accessibility_21_moran.p_sim],
        'n_permutations': [accessibility_21_moran.permutations]
    })
    moran_results_accessibility21.to_csv(outputs["OLS_accessibility_2021_moran"], index=False)

    # multivariate regression analysis
    multiv_accessibility_summary_table_21, multiv_accessibility21_residuals, multiv_accessibility21_moran = OLS_analysis_multivariate_moran(Regression_21_24, dependent_variable, multiv_independent_variables)
    # save the summary table
    multiv_accessibility_summary_table_21.to_csv(outputs["multiv_OLS_accessibility_2021"], index=False)

    # save the residuals
    # Create a copy of the cleaned data used for modeling
    multiv_accessibility21_residuals_df = Regression_21_24.copy().reset_index(drop=True)
    # Convert residuals to a pandas Series with the same index
    residuals_series = pd.Series(multiv_accessibility21_residuals, name='residuals')
    # Add residuals column to the DataFrame
    multiv_accessibility21_residuals_df['residuals'] = residuals_series
    multiv_accessibility21_residuals_df[["residuals"]].to_csv(outputs["multiv_OLS_accessibility_2021_residuals"], index=False)

    # save the moran's I results
    # Create a DataFrame with Moran's I results
    moran_results_accessibility21 = pd.DataFrame({
        'Moran_I': [multiv_accessibility21_moran.I],
        'Expected_I': [multiv_accessibility21_moran.EI],
        'Variance': [multiv_accessibility21_moran.VI_norm],
        'z_score': [multiv_accessibility21_moran.z_norm],
        'p_value': [multiv_accessibility21_moran.p_norm],
        'p_sim': [multiv_accessibility21_moran.p_sim],
        'n_permutations': [multiv_accessibility21_moran.permutations]
    })

    moran_results_accessibility21.to_csv(outputs["multiv_OLS_accessibility_2021_moran"], index=False)

    # __________________________________________________________________________________________________________________
    # OLS for ACCESSIBILITY 2024
    dependent_variable = "acc_24"
    independent_variables = ["s-ASG_ABC1",        # Share of HH in social grade AB and C1
                           "s-ASG_C2DE",       # Share of HH in social grade C2 and DE
                           "s-D3+",             # Share of HH deprived in 3+ dimensions
                           "s-HHcars01",      # Share of HH with 0 or 1 car
                           "s-HHcars2+",       # Share of HH with 2+ cars
                           "s-HHT_own",       # share of HH owning outright + mortgage + shared ownership
                           "s-HHT_rent",     # share of HH renting
                           "s-semi-det", # Share of HH living in detached and semidetached houses
                           "s-flat",        # Share of HH living in flats
                           "s-terr-oth",       # Share of HH living in terraced & other houses
                           #"Med_HP_21",      # Median house prices 2021 (December)
                           "Med_HP_23",       # Median house prices 2023 (March)
                           "Pop_dens",       # Population density (thousands)
                           #"RoadKmDen",         # Road network (km) density
                           "POI_dens"]          # POI density
                           #"job_th_21",         # Thousands of jobs per LSOA in 2021
                           #"job_th_23"]         # Thousands of jobs per LSOA in 2023

    multiv_independent_variables = [  # "s-ASG_ABC1",  # Share of HH in social grade AB and C1
                                "s-ASG_C2DE",  # Share of HH in social grade C2 and DE
                                "s-D3+",  # Share of HH deprived in 3+ dimensions
                                "s-HHcars01",  # Share of HH with 0 or 1 car
                                # "s-HHcars2+",  # Share of HH with 2+ cars
                                # "s-HHT_own",  # share of HH owning outright + mortgage + shared ownership
                                "s-HHT_rent",  # share of HH renting
                                # "s-semi-det",  # Share of HH living in detached and semidetached houses
                                "s-flat",  # Share of HH living in flats
                                "s-terr-oth",  # Share of HH living in terraced & other houses
                                #"Med_HP_21",  # Median house prices 2021 (December)
                                "Med_HP_23",       # Median house prices 2023 (March)
                                "Pop_dens",  # Population density (thousands)
                                # "RoadKmDen",         # Road network (km) density
                                "POI_dens"]  # POI density
                            # "job_th_21"]         # Thousands of jobs per LSOA in 2021
                            # "job_th_23"]         # Thousands of jobs per LSOA in 2023


    # Univariate regression analysis
    #accessibility_summary_table_24 = OLS_analysis(Regression_21_24, dependent_variable, independent_variables)
    accessibility_summary_table_24, accessibility_24_residuals, accessibility_24_moran = OLS_analysis_univariate_moran(Regression_21_24, dependent_variable, independent_variables)

    # save the summary table
    accessibility_summary_table_24.to_csv(outputs["OLS_accessibility_2024"], index=False)

    # save the residuals
    # Create a copy of the cleaned data used for modeling
    accessibility_24_residuals_df = Regression_21_24.copy().reset_index(drop=True)
    # Convert residuals to a pandas Series with the same index
    residuals_series = pd.Series(accessibility_24_residuals, name='residuals')
    # Add residuals column to the DataFrame
    accessibility_24_residuals_df['residuals'] = residuals_series
    accessibility_24_residuals_df[["residuals"]].to_csv(outputs["OLS_accessibility_2024_residuals"], index=False)

    # save the moran's I results
    # Create a DataFrame with Moran's I results
    moran_results_accessibility24 = pd.DataFrame({
        'Moran_I': [accessibility_24_moran.I],
        'Expected_I': [accessibility_24_moran.EI],
        'Variance': [accessibility_24_moran.VI_norm],
        'z_score': [accessibility_24_moran.z_norm],
        'p_value': [accessibility_24_moran.p_norm],
        'p_sim': [accessibility_24_moran.p_sim],
        'n_permutations': [accessibility_24_moran.permutations]
    })
    moran_results_accessibility24.to_csv(outputs["OLS_accessibility_2024_moran"], index=False)

    # multivariate regression analysis
    multiv_accessibility_summary_table_24, multiv_accessibility24_residuals, multiv_accessibility24_moran = OLS_analysis_multivariate_moran(Regression_21_24, dependent_variable, multiv_independent_variables)
    # save the summary table
    multiv_accessibility_summary_table_24.to_csv(outputs["multiv_OLS_accessibility_2024"], index=False)

    # save the residuals
    # Create a copy of the cleaned data used for modeling
    multiv_accessibility24_residuals_df = Regression_21_24.copy().reset_index(drop=True)
    # Convert residuals to a pandas Series with the same index
    residuals_series = pd.Series(multiv_accessibility24_residuals, name='residuals')
    # Add residuals column to the DataFrame
    multiv_accessibility24_residuals_df['residuals'] = residuals_series
    multiv_accessibility24_residuals_df[["residuals"]].to_csv(outputs["multiv_OLS_accessibility_2024_residuals"], index=False)

    # save the moran's I results
    # Create a DataFrame with Moran's I results
    moran_results_accessibility24 = pd.DataFrame({
        'Moran_I': [multiv_accessibility24_moran.I],
        'Expected_I': [multiv_accessibility24_moran.EI],
        'Variance': [multiv_accessibility24_moran.VI_norm],
        'z_score': [multiv_accessibility24_moran.z_norm],
        'p_value': [multiv_accessibility24_moran.p_norm],
        'p_sim': [multiv_accessibility24_moran.p_sim],
        'n_permutations': [multiv_accessibility24_moran.permutations]
    })
    moran_results_accessibility24.to_csv(outputs["multiv_OLS_accessibility_2024_moran"], index=False)

    # __________________________________________________________________________________________________________________
    # OLS for ACCESSIBILITY DIFFERENCE 2024-2021
    dependent_variable = "acc_diff"
    independent_variables = ["s-ASG_ABC1",        # Share of HH in social grade AB and C1
                           "s-ASG_C2DE",       # Share of HH in social grade C2 and DE
                           "s-D3+",             # Share of HH deprived in 3+ dimensions
                           "s-HHcars01",      # Share of HH with 0 or 1 car
                           "s-HHcars2+",       # Share of HH with 2+ cars
                           "s-HHT_own",       # share of HH owning outright + mortgage + shared ownership
                           "s-HHT_rent",     # share of HH renting
                           "s-semi-det", # Share of HH living in detached and semidetached houses
                           "s-flat",        # Share of HH living in flats
                           "s-terr-oth",       # Share of HH living in terraced & other houses
                           #"Med_HP_21",      # Median house prices 2021 (December)
                           "Med_HP_23",       # Median house prices 2023 (March)
                           "Pop_dens",       # Population density (thousands)
                           #"RoadKmDen",         # Road network (km) density
                           "POI_dens"]          # POI density
                           #"job_th_21",         # Thousands of jobs per LSOA in 2021
                           #"job_th_23"]         # Thousands of jobs per LSOA in 2023

    multiv_independent_variables = [  # "s-ASG_ABC1",  # Share of HH in social grade AB and C1
                            "s-ASG_C2DE",  # Share of HH in social grade C2 and DE
                            "s-D3+",  # Share of HH deprived in 3+ dimensions
                            "s-HHcars01",  # Share of HH with 0 or 1 car
                            # "s-HHcars2+",  # Share of HH with 2+ cars
                            # "s-HHT_own",  # share of HH owning outright + mortgage + shared ownership
                            "s-HHT_rent",  # share of HH renting
                            # "s-semi-det",  # Share of HH living in detached and semidetached houses
                            "s-flat",  # Share of HH living in flats
                            "s-terr-oth",  # Share of HH living in terraced & other houses
                            # "Med_HP_21",  # Median house prices 2021 (December)
                            "Med_HP_23",  # Median house prices 2023 (March)
                            "Pop_dens",  # Population density (thousands)
                            # "RoadKmDen",         # Road network (km) density
                            "POI_dens"]  # POI density
                        # "job_th_21"]         # Thousands of jobs per LSOA in 2021
                        # "job_th_23"]         # Thousands of jobs per LSOA in 2023

    # Univariate regression analysis
    #acc_diff_summary_table_24_21 = OLS_analysis(Regression_21_24, dependent_variable, independent_variables)
    acc_diff_summary_table_24_21, acc_diff_24_21_residuals, acc_diff_24_21_moran = OLS_analysis_univariate_moran(Regression_21_24, dependent_variable, independent_variables)

    # save the summary table
    acc_diff_summary_table_24_21.to_csv(outputs["OLS_diff_accessibility_21_24"], index=False)

    # save the residuals
    # Create a copy of the cleaned data used for modeling
    acc_diff_24_21_residuals_df = Regression_21_24.copy().reset_index(drop=True)
    # Convert residuals to a pandas Series with the same index
    residuals_series = pd.Series(acc_diff_24_21_residuals, name='residuals')
    # Add residuals column to the DataFrame
    acc_diff_24_21_residuals_df['residuals'] = residuals_series
    acc_diff_24_21_residuals_df[["residuals"]].to_csv(outputs["OLS_diff_accessibility_21_24_residuals"], index=False)

    # save the moran's I results
    # Create a DataFrame with Moran's I results
    moran_results_acc_diff24_21 = pd.DataFrame({
        'Moran_I': [acc_diff_24_21_moran.I],
        'Expected_I': [acc_diff_24_21_moran.EI],
        'Variance': [acc_diff_24_21_moran.VI_norm],
        'z_score': [acc_diff_24_21_moran.z_norm],
        'p_value': [acc_diff_24_21_moran.p_norm],
        'p_sim': [acc_diff_24_21_moran.p_sim],
        'n_permutations': [acc_diff_24_21_moran.permutations]
    })
    moran_results_acc_diff24_21.to_csv(outputs["OLS_diff_accessibility_21_24_moran"], index=False)

    # multivariate regression analysis
    multiv_acc_diff_summary_table_24_21, multiv_acc_diff24_21_residuals, multiv_acc_diff24_21_moran = OLS_analysis_multivariate_moran(Regression_21_24, dependent_variable, multiv_independent_variables)
    # save the summary table
    multiv_acc_diff_summary_table_24_21.to_csv(outputs["multiv_OLS_diff_accessibility_21_24"], index=False)

    # save the residuals
    # Create a copy of the cleaned data used for modeling
    multiv_acc_diff24_21_residuals_df = Regression_21_24.copy().reset_index(drop=True)
    # Convert residuals to a pandas Series with the same index
    residuals_series = pd.Series(multiv_acc_diff24_21_residuals, name='residuals')
    # Add residuals column to the DataFrame
    multiv_acc_diff24_21_residuals_df['residuals'] = residuals_series
    multiv_acc_diff24_21_residuals_df[["residuals"]].to_csv(outputs["multiv_OLS_diff_accessibility_21_24_residuals"], index=False)

    # save the moran's I results
    # Create a DataFrame with Moran's I results
    moran_results_acc_diff24_21 = pd.DataFrame({
        'Moran_I': [multiv_acc_diff24_21_moran.I],
        'Expected_I': [multiv_acc_diff24_21_moran.EI],
        'Variance': [multiv_acc_diff24_21_moran.VI_norm],
        'z_score': [multiv_acc_diff24_21_moran.z_norm],
        'p_value': [multiv_acc_diff24_21_moran.p_norm],
        'p_sim': [multiv_acc_diff24_21_moran.p_sim],
        'n_permutations': [multiv_acc_diff24_21_moran.permutations]
    })
    moran_results_acc_diff24_21.to_csv(outputs["multiv_OLS_diff_accessibility_21_24_moran"], index=False)

    # __________________________________________________________________________________________________________________
    # OLS for EVCI SUPPLY IMPROVEMENT
    dependent_variable = "s_impr"
    independent_variables = ["s-ASG_ABC1",        # Share of HH in social grade AB and C1
                           "s-ASG_C2DE",       # Share of HH in social grade C2 and DE
                           "s-D3+",             # Share of HH deprived in 3+ dimensions
                           "s-HHcars01",      # Share of HH with 0 or 1 car
                           "s-HHcars2+",       # Share of HH with 2+ cars
                           "s-HHT_own",       # share of HH owning outright + mortgage + shared ownership
                           "s-HHT_rent",     # share of HH renting
                           "s-semi-det", # Share of HH living in detached and semidetached houses
                           "s-flat",        # Share of HH living in flats
                           "s-terr-oth",       # Share of HH living in terraced & other houses
                           #"Med_HP_21",      # Median house prices 2021 (December)
                           "Med_HP_23",       # Median house prices 2023 (March)
                           "Pop_dens",       # Population density (thousands)
                           #"RoadKmDen",         # Road network (km) density
                           "POI_dens"]          # POI density
                           #"job_th_21",         # Thousands of jobs per LSOA in 2021
                           #"job_th_23"]         # Thousands of jobs per LSOA in 2023

    multiv_independent_variables = [  # "s-ASG_ABC1",  # Share of HH in social grade AB and C1
                            "s-ASG_C2DE",  # Share of HH in social grade C2 and DE
                            "s-D3+",  # Share of HH deprived in 3+ dimensions
                            "s-HHcars01",  # Share of HH with 0 or 1 car
                            # "s-HHcars2+",  # Share of HH with 2+ cars
                            # "s-HHT_own",  # share of HH owning outright + mortgage + shared ownership
                            "s-HHT_rent",  # share of HH renting
                            # "s-semi-det",  # Share of HH living in detached and semidetached houses
                            "s-flat",  # Share of HH living in flats
                            "s-terr-oth",  # Share of HH living in terraced & other houses
                            # "Med_HP_21",  # Median house prices 2021 (December)
                            "Med_HP_23",  # Median house prices 2023 (March)
                            "Pop_dens",  # Population density (thousands)
                            # "RoadKmDen",         # Road network (km) density
                            "POI_dens"]  # POI density
                        # "job_th_21"]         # Thousands of jobs per LSOA in 2021
                        # "job_th_23"]         # Thousands of jobs per LSOA in 2023

    # Univariate regression analysis
    #supp_impr_summary_table_24_21 = OLS_analysis(Regression_21_24, dependent_variable, independent_variables)
    supp_impr_summary_table_24_21, supp_impr_24_21_residuals, supp_impr_24_21_moran = OLS_analysis_univariate_moran(Regression_21_24, dependent_variable, independent_variables)

    # save the summary table
    supp_impr_summary_table_24_21.to_csv(outputs["OLS_supply_improvement"], index=False)

    # save the residuals
    # Create a copy of the cleaned data used for modeling
    supp_impr_24_21_residuals_df = Regression_21_24.copy().reset_index(drop=True)
    # Convert residuals to a pandas Series with the same index
    residuals_series = pd.Series(supp_impr_24_21_residuals, name='residuals')
    # Add residuals column to the DataFrame
    supp_impr_24_21_residuals_df['residuals'] = residuals_series
    supp_impr_24_21_residuals_df[["residuals"]].to_csv(outputs["OLS_supply_improvement_residuals"], index=False)

    # save the moran's I results
    # Create a DataFrame with Moran's I results
    moran_results_supp_impr24_21 = pd.DataFrame({
        'Moran_I': [supp_impr_24_21_moran.I],
        'Expected_I': [supp_impr_24_21_moran.EI],
        'Variance': [supp_impr_24_21_moran.VI_norm],
        'z_score': [supp_impr_24_21_moran.z_norm],
        'p_value': [supp_impr_24_21_moran.p_norm],
        'p_sim': [supp_impr_24_21_moran.p_sim],
        'n_permutations': [supp_impr_24_21_moran.permutations]
    })
    moran_results_supp_impr24_21.to_csv(outputs["OLS_supply_improvement_moran"], index=False)

    # multivariate regression analysis
    multiv_supp_impr_summary_table_24_21, multiv_supp_impr_24_21_residuals, multiv_supp_impr_24_21_moran = OLS_analysis_multivariate_moran(Regression_21_24, dependent_variable, multiv_independent_variables)
    # save the summary table
    multiv_supp_impr_summary_table_24_21.to_csv(outputs["multiv_OLS_supply_improvement"], index=False)

    # save the residuals
    # Create a copy of the cleaned data used for modeling
    multiv_supp_impr_24_21_residuals_df = Regression_21_24.copy().reset_index(drop=True)
    # Convert residuals to a pandas Series with the same index
    residuals_series = pd.Series(multiv_supp_impr_24_21_residuals, name='residuals')
    # Add residuals column to the DataFrame
    multiv_supp_impr_24_21_residuals_df['residuals'] = residuals_series
    multiv_supp_impr_24_21_residuals_df[["residuals"]].to_csv(outputs["multiv_OLS_supply_improvement_residuals"], index=False)

    # save the moran's I results
    # Create a DataFrame with Moran's I results
    moran_results_supp_impr_24_21 = pd.DataFrame({
        'Moran_I': [multiv_supp_impr_24_21_moran.I],
        'Expected_I': [multiv_supp_impr_24_21_moran.EI],
        'Variance': [multiv_supp_impr_24_21_moran.VI_norm],
        'z_score': [multiv_supp_impr_24_21_moran.z_norm],
        'p_value': [multiv_supp_impr_24_21_moran.p_norm],
        'p_sim': [multiv_supp_impr_24_21_moran.p_sim],
        'n_permutations': [multiv_supp_impr_24_21_moran.permutations]
    })
    moran_results_supp_impr_24_21.to_csv(outputs["multiv_OLS_supply_improvement_moran"], index=False)

    # __________________________________________________________________________________________________________________
    # OLS for EVCI DEMAND IMPROVEMENT
    dependent_variable = "d_impr"
    independent_variables = ["s-ASG_ABC1",        # Share of HH in social grade AB and C1
                           "s-ASG_C2DE",       # Share of HH in social grade C2 and DE
                           "s-D3+",             # Share of HH deprived in 3+ dimensions
                           "s-HHcars01",      # Share of HH with 0 or 1 car
                           "s-HHcars2+",       # Share of HH with 2+ cars
                           "s-HHT_own",       # share of HH owning outright + mortgage + shared ownership
                           "s-HHT_rent",     # share of HH renting
                           "s-semi-det", # Share of HH living in detached and semidetached houses
                           "s-flat",        # Share of HH living in flats
                           "s-terr-oth",       # Share of HH living in terraced & other houses
                           #"Med_HP_21",      # Median house prices 2021 (December)
                           "Med_HP_23",       # Median house prices 2023 (March)
                           "Pop_dens",       # Population density (thousands)
                           #"RoadKmDen",         # Road network (km) density
                           "POI_dens"]          # POI density
                           #"job_th_21",         # Thousands of jobs per LSOA in 2021
                           #"job_th_23"]         # Thousands of jobs per LSOA in 2023

    multiv_independent_variables = [  # "s-ASG_ABC1",  # Share of HH in social grade AB and C1
                        "s-ASG_C2DE",  # Share of HH in social grade C2 and DE
                        "s-D3+",  # Share of HH deprived in 3+ dimensions
                        "s-HHcars01",  # Share of HH with 0 or 1 car
                        # "s-HHcars2+",  # Share of HH with 2+ cars
                        # "s-HHT_own",  # share of HH owning outright + mortgage + shared ownership
                        "s-HHT_rent",  # share of HH renting
                        # "s-semi-det",  # Share of HH living in detached and semidetached houses
                        "s-flat",  # Share of HH living in flats
                        "s-terr-oth",  # Share of HH living in terraced & other houses
                        # "Med_HP_21",  # Median house prices 2021 (December)
                        "Med_HP_23",  # Median house prices 2023 (March)
                        "Pop_dens",  # Population density (thousands)
                        # "RoadKmDen",         # Road network (km) density
                        "POI_dens"]  # POI density
                    # "job_th_21"]         # Thousands of jobs per LSOA in 2021
                    # "job_th_23"]         # Thousands of jobs per LSOA in 2023

    # Univariate regression analysis

    #dem_impr_summary_table_24_21 = OLS_analysis(Regression_21_24, dependent_variable, independent_variables)
    dem_impr_summary_table_24_21, dem_impr_24_21_residuals, dem_impr_24_21_moran = OLS_analysis_univariate_moran(Regression_21_24, dependent_variable, independent_variables)

    # save the summary table
    dem_impr_summary_table_24_21.to_csv(outputs["OLS_demand_improvement"], index=False)

    # save the residuals
    # Create a copy of the cleaned data used for modeling
    dem_impr_24_21_residuals_df = Regression_21_24.copy().reset_index(drop=True)
    # Convert residuals to a pandas Series with the same index
    residuals_series = pd.Series(dem_impr_24_21_residuals, name='residuals')
    # Add residuals column to the DataFrame
    dem_impr_24_21_residuals_df['residuals'] = residuals_series
    dem_impr_24_21_residuals_df[["residuals"]].to_csv(outputs["OLS_demand_improvement_residuals"], index=False)

    # save the moran's I results
    # Create a DataFrame with Moran's I results
    moran_results_dem_impr24_21 = pd.DataFrame({
        'Moran_I': [dem_impr_24_21_moran.I],
        'Expected_I': [dem_impr_24_21_moran.EI],
        'Variance': [dem_impr_24_21_moran.VI_norm],
        'z_score': [dem_impr_24_21_moran.z_norm],
        'p_value': [dem_impr_24_21_moran.p_norm],
        'p_sim': [dem_impr_24_21_moran.p_sim],
        'n_permutations': [dem_impr_24_21_moran.permutations]
    })
    moran_results_dem_impr24_21.to_csv(outputs["OLS_demand_improvement_moran"], index=False)

    # multivariate regression analysis
    multiv_dem_impr_summary_table_24_21, multiv_dem_impr24_21_residuals, multiv_dem_impr24_21_moran = OLS_analysis_multivariate_moran(Regression_21_24, dependent_variable, multiv_independent_variables)
    # save the summary table
    multiv_dem_impr_summary_table_24_21.to_csv(outputs["multiv_OLS_demand_improvement"], index=False)

    # save the residuals
    # Create a copy of the cleaned data used for modeling
    multiv_dem_impr24_21_residuals_df = Regression_21_24.copy().reset_index(drop=True)
    # Convert residuals to a pandas Series with the same index
    residuals_series = pd.Series(multiv_dem_impr24_21_residuals, name='residuals')
    # Add residuals column to the DataFrame
    multiv_dem_impr24_21_residuals_df['residuals'] = residuals_series
    multiv_dem_impr24_21_residuals_df[["residuals"]].to_csv(outputs["multiv_OLS_demand_improvement_residuals"], index=False)

    # save the moran's I results
    # Create a DataFrame with Moran's I results
    moran_results_dem_impr24_21 = pd.DataFrame({
        'Moran_I': [multiv_dem_impr24_21_moran.I],
        'Expected_I': [multiv_dem_impr24_21_moran.EI],
        'Variance': [multiv_dem_impr24_21_moran.VI_norm],
        'z_score': [multiv_dem_impr24_21_moran.z_norm],
        'p_value': [multiv_dem_impr24_21_moran.p_norm],
        'p_sim': [multiv_dem_impr24_21_moran.p_sim],
        'n_permutations': [multiv_dem_impr24_21_moran.permutations]
    })
    moran_results_dem_impr24_21.to_csv(outputs["multiv_OLS_demand_improvement_moran"], index=False)

########################################################################################################################
# calculate the LM-lag, LM-error, Robust LM-lag, and Robust LM-error tests

LM_lag_error_test_flag = False

if LM_lag_error_test_flag == True:
    print("Warning: do not run this together with OLS analysis, as it will break due to overwriting the Regression_21_24 dataframe")
    London_LSOA_centroids = gpd.read_file(inputs["London_LSOA_centroids"])
    # Now use the polygons geometry for the Regression_21_24 dataframe
    Regression_21_24 = Regression_21_24.merge(London_LSOA_centroids, on="LSOA21CD", how="outer")
    # Now turn the Regression_21_24 dataframe into a geodataframe
    Regression_21_24 = gpd.GeoDataFrame(Regression_21_24)

    # Rename columns
    Regression_21_24.rename(columns={"LSOA21NM_x": "LSOA21NM"}, inplace=True)
    # Drop extra columns
    Regression_21_24.drop(columns=["LSOA21NM_y", "GlobalID"], inplace=True)

    # Remove NaNs
    Regression_21_24 = Regression_21_24.dropna()

    # Turn the geometry column into a x and y column
    Regression_21_24["x"] = Regression_21_24.centroid.x
    Regression_21_24["y"] = Regression_21_24.centroid.y

    dep_vars_test = ["acc_21", "acc_24", "acc_diff", "s_impr", "d_impr"]

    indep_vars_test = [["s-ASG_C2DE", "s-D3+", "s-HHcars01", "s-HHT_rent", "s-flat", "s-terr-oth", "Med_HP_21", "Pop_dens", "POI_dens"],
                       ["s-ASG_C2DE", "s-D3+", "s-HHcars01", "s-HHT_rent", "s-flat", "s-terr-oth", "Med_HP_23", "Pop_dens", "POI_dens"],
                       ["s-ASG_C2DE", "s-D3+", "s-HHcars01", "s-HHT_rent", "s-flat", "s-terr-oth", "Med_HP_23", "Pop_dens", "POI_dens"],
                       ["s-ASG_C2DE", "s-D3+", "s-HHcars01", "s-HHT_rent", "s-flat", "s-terr-oth", "Med_HP_23", "Pop_dens", "POI_dens"],
                       ["s-ASG_C2DE", "s-D3+", "s-HHcars01", "s-HHT_rent", "s-flat", "s-terr-oth", "Med_HP_23", "Pop_dens", "POI_dens"]]

    lm_summary = lm_tests_multiple_models(Regression_21_24, dep_vars_test, indep_vars_test)

    # add statistical significance to the summary table (p < 0.05)
    lm_summary["LM Lag Sig"] = lm_summary["LM Lag p-value"].apply(lambda x: "*" if x < 0.05 else "")
    lm_summary["LM Error Sig"] = lm_summary["LM Error p-value"].apply(lambda x: "*" if x < 0.05 else "")
    lm_summary["Robust LM Lag Sig"] = lm_summary["Robust LM Lag p-value"].apply(lambda x: "*" if x < 0.05 else "")
    lm_summary["Robust LM Error Sig"] = lm_summary["Robust LM Error p-value"].apply(lambda x: "*" if x < 0.05 else "")

    # reorder the columns so that stat significance is next to the p-values
    lm_summary = lm_summary[["Dependent Variable", "Independent Variables",
                             "LM Lag", "LM Lag p-value", "LM Lag Sig",
                             "LM Error", "LM Error p-value", "LM Error Sig",
                             "Robust LM Lag", "Robust LM Lag p-value", "Robust LM Lag Sig",
                             "Robust LM Error", "Robust LM Error p-value", "Robust LM Error Sig"]]

    # print the whole summary table, not just a preview:
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.width', None)  # No limit on width
    pd.set_option('display.max_colwidth', None)  # No limit on column width
    print(lm_summary)

    # save the summary table
    lm_summary.to_csv(outputs["LM_lag_error_test"], index=False)

########################################################################################################################
# Spatial lag model (SLM)
# SLM: if I suspect spatial dependence in the outcome — e.g., accessibility in one borough depends on accessibility in neighboring boroughs.

SLM_flag = False

if SLM_flag == True:
    print(
        "Warning: do not run this together with OLS analysis, as it will break due to overwriting the Regression_21_24 dataframe")
    London_LSOA_centroids = gpd.read_file(inputs["London_LSOA_centroids"])
    # Now use the polygons geometry for the Regression_21_24 dataframe
    Regression_21_24 = Regression_21_24.merge(London_LSOA_centroids, on="LSOA21CD", how="outer")
    # Now turn the Regression_21_24 dataframe into a geodataframe
    Regression_21_24 = gpd.GeoDataFrame(Regression_21_24)

    # Rename columns
    Regression_21_24.rename(columns={"LSOA21NM_x": "LSOA21NM"}, inplace=True)
    # Drop extra columns
    Regression_21_24.drop(columns=["LSOA21NM_y", "GlobalID"], inplace=True)

    # Remove NaNs
    Regression_21_24 = Regression_21_24.dropna()

    # Turn the geometry column into a x and y column
    Regression_21_24["x"] = Regression_21_24.centroid.x
    Regression_21_24["y"] = Regression_21_24.centroid.y

    # Run SLM for accessibility_2021 and accessibility_difference_2021-2024
    slm_results_acc_21 = run_slm(Regression_21_24, "acc_21", ["s-ASG_C2DE", "s-D3+", "s-HHcars01", "s-HHT_rent", "s-flat", "s-terr-oth", "Med_HP_21", "Pop_dens", "POI_dens"])
    slm_results_acc_diff = run_slm(Regression_21_24, "acc_diff", ["s-ASG_C2DE", "s-D3+", "s-HHcars01", "s-HHT_rent", "s-flat", "s-terr-oth", "Med_HP_23", "Pop_dens", "POI_dens"])
    sdm_results_acc_24 = run_slm(Regression_21_24, "acc_24", ["s-ASG_C2DE", "s-D3+", "s-HHcars01", "s-HHT_rent", "s-flat", "s-terr-oth", "Med_HP_23", "Pop_dens", "POI_dens"])
    sdm_results_supp_impr = run_slm(Regression_21_24, "s_impr", ["s-ASG_C2DE", "s-D3+", "s-HHcars01", "s-HHT_rent", "s-flat", "s-terr-oth", "Med_HP_23", "Pop_dens", "POI_dens"])
    sdm_results_dem_impr = run_slm(Regression_21_24, "d_impr", ["s-ASG_C2DE", "s-D3+", "s-HHcars01", "s-HHT_rent", "s-flat", "s-terr-oth", "Med_HP_23", "Pop_dens", "POI_dens"])


    # Save the results to CSV, each result in a separate file
    slm_results_acc_21.to_csv(outputs["SLM_acc_21"], index=False)
    slm_results_acc_diff.to_csv(outputs["SLM_diff_acc_21_24"], index=False)
    sdm_results_acc_24.to_csv(outputs["SLM_acc_24"], index=False)
    sdm_results_supp_impr.to_csv(outputs["SLM_supply_impr"], index=False)
    sdm_results_dem_impr.to_csv(outputs["SLM_demand_impr"], index=False)


########################################################################################################################
# GWR analysis
# reference code: https://github.com/urschrei/Geopython/blob/master/geographically_weighted_regression.ipynb

GWR_flag = False
# Normalisation options:
normalise_dependent_variables = False
normalise_independent_variables = False

def lsoa_boundaries(ax):
    """ to plot boundaries on maps """
    return lsoa_gdf.to_crs('epsg:27700').plot(
            ax=ax,
            linewidth=.5,
            color='#555555',
            edgecolor='w',
            alpha=.5,
            zorder=1)

# For GWR analysis:
# dependent variables: take the inverse hyperbolic sine transformation (because they have 0 and negative values)
#Regression_21_24["accessibility_21_arcsinh"] = np.arcsinh(Regression_21_24["accessibility_21"])
#Regression_21_24["accessibility_24_arcsinh"] = np.arcsinh(Regression_21_24["accessibility_24"])
#Regression_21_24["acc_diff_24_21_arcsinh"] = np.arcsinh(Regression_21_24["acc_diff_24_21"])


if GWR_flag == True:
    London_LSOA_centroids = gpd.read_file(inputs["London_LSOA_centroids"])
    # Now use the polygons geometry for the Regression_21_24 dataframe
    Regression_21_24 = Regression_21_24.merge(London_LSOA_centroids, on="LSOA21CD", how="outer")
    # Now turn the Regression_21_24 dataframe into a geodataframe
    Regression_21_24 = gpd.GeoDataFrame(Regression_21_24)

    # Rename columns
    Regression_21_24.rename(columns={"LSOA21NM_x": "LSOA21NM"}, inplace=True)
    # Drop extra columns
    Regression_21_24.drop(columns=["LSOA21NM_y", "GlobalID"], inplace=True)

    # Remove NaNs
    Regression_21_24 = Regression_21_24.dropna()

    # Turn the geometry column into a x and y column
    Regression_21_24["x"] = Regression_21_24.centroid.x
    Regression_21_24["y"] = Regression_21_24.centroid.y

    # categorise the variables in dependent and independent variables and save the categorisation into two lists:
    cat_dep_variables = [
                         #"EVCI2021",                              # EVCI 2021
                         #"y2021Q4",                               # EV licensing 2021
                         #"EVCI2024",                              # EVCI 2024
                         #"y2024Q2",                               # EV licensing 2024
                         "acc_21",                                # Accessibility 2021
                         "acc_24",                                # Accessibility 2024
                         "acc_diff",                              # Accessibility difference 2024 - 2021
                         #"accessibility_21_arcsinh",              # Accessibility 2021 arcsinh
                         #"accessibility_24_arcsinh",              # Accessibility 2024 arcsinh
                         #"acc_diff_24_21_arcsinh",                # Accessibility difference 2024 - 2021 arcsinh
                         "s_impr",                                # EVCI (supply) improvement
                         #"s_imp_rate",                             # EVCI (supply) improvement rate
                         #"EVCI_improvement_rate_arcsinh",         # EVCI (supply) improvement rate
                         "d_impr"]                                # EV licensing (demand) improvement
                         #"d_imp_rate"]                             # EV licensing (demand) improvement rate
                         #"EV_licensing_improvement_rate_arcsinh"] # EV licensing (demand) improvement rat

    # if any of the dependent variables have nans, drop them
    # check if there are any nans in the dependent variables, if so, print a warning
    if Regression_21_24[cat_dep_variables].isnull().values.any():
        print("---------------------------------------------------------------------")
        print("WARNING: there are NaNs in the dependent variables. Dropping them...")
        print("---------------------------------------------------------------------")
        Regression_21_24 = Regression_21_24.dropna(subset=cat_dep_variables)

    cat_indep_variables = [
                           #"s-ASG_ABC1",        # Share of HH in social grade AB and C1
                           "s-ASG_C2DE",       # Share of HH in social grade C2 and DE
                           "s-D3+",             # Share of HH deprived in 3+ dimensions
                           "s-HHcars01",      # Share of HH with 0 or 1 car
                           #"s-HHcars2+",       # Share of HH with 2+ cars
                           #"s-HHT_own",       # share of HH owning outright + mortgage + shared ownership
                           "s-HHT_rent",     # share of HH renting
                           #"s-semi-det", # Share of HH living in detached and semidetached houses
                           "s-flat",        # Share of HH living in flats
                           "s-terr-oth",       # Share of HH living in terraced & other houses
                           #"Med_HP_21",      # Median house prices 2021 (December)
                           "Med_HP_23",       # Median house prices 2023 (March)
                           "Pop_dens",       # Population density (thousands)
                           #"RoadKmDen",         # Road network (km) density
                           "POI_dens"]          # POI density
                           #"job_th_21",         # Thousands of jobs per LSOA in 2021
                           #"job_th_23"]         # Thousands of jobs per LSOA in 2023

    # use min max scaling for all the variables
    for i in cat_indep_variables:
        Regression_21_24[i] = (Regression_21_24[i] - Regression_21_24[i].min()) / (
                Regression_21_24[i].max() - Regression_21_24[i].min())

    for i in cat_dep_variables:
        Regression_21_24[i] = (Regression_21_24[i] - Regression_21_24[i].min()) / (
                Regression_21_24[i].max() - Regression_21_24[i].min())

    # plot min and max values of the dependent variables
    for dep in cat_indep_variables:
        print("Min and max values of the dependent variable: ", dep)
        print("Min: ", Regression_21_24[dep].min())
        print("Max: ", Regression_21_24[dep].max())
        print()

    if normalise_dependent_variables == True:
        for i in cat_dep_variables:
            Regression_21_24[i] = (Regression_21_24[i] - Regression_21_24[i].min()) / (
                    Regression_21_24[i].max() - Regression_21_24[i].min())

    if normalise_independent_variables == True:
        for i in cat_indep_variables:
            Regression_21_24[i] = (Regression_21_24[i] - Regression_21_24[i].min()) / (
                    Regression_21_24[i].max() - Regression_21_24[i].min())


    # arrange endog (y) as a column vector, i.e. an m x 1 array
    for dep in cat_dep_variables:
        print("GWR analysis for: ", dep)
        endog = Regression_21_24[dep].values.reshape(-1, 1)
        # exog (X) is an m x n array, where m is the number of rows, and n is the number of regressors
        exog = Regression_21_24[cat_indep_variables].values

        # Python 3 zip returns an iterable
        coords = list(zip(Regression_21_24.x.values, Regression_21_24.y.values))

        # Instantiate bandwidth selection class - bisquare NN (adaptive)
        bw = Sel_BW(
            coords,
            endog,
            exog,
            kernel='bisquare', fixed=False)

        # Find optimal bandwidth by minimizing AICc using golden section search algorithm
        bw = Sel_BW(coords, endog, exog).search(criterion='AICc')
        print("GWR Bandwith: ", bw)

        # Instantiate GWR model and estimate parameters and diagnostics
        model = GWR(
            coords,
            endog,
            exog,
            bw,
            family=Gaussian(),
            fixed=False,
            kernel='gaussian')

        results = model.fit()

        # Map local R-square values (a weighted R-square at each observation location)
        fig, ax = plt.subplots(1,
                               figsize=(8, 8),
                               dpi=100,
                               subplot_kw=dict(aspect='equal'))

        # add local R2 to df
        Regression_21_24['localR2'] = results.localR2
        vmin, vmax = np.min(Regression_21_24['localR2']), np.max(Regression_21_24['localR2'])

        Regression_21_24.plot(
            'localR2',
            markersize=10.,
            edgecolor='#555555',
            linewidths=.25,
            vmin=vmin,
            vmax=vmax,
            cmap='viridis',
            ax=ax,
            zorder=2)

        # impose LSOA boundaries
        lsoa_boundaries(ax)

        ax.set_title('Local R-Squared. Dep. var: ' + dep)
        fig = ax.get_figure()
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap='viridis')
        sm._A = []
        #fig.colorbar(sm)
        cbar = fig.colorbar(sm, ax=ax) # This ensures the colorbar is linked to the correct axis

        _ = ax.axis('off')

        # Save the figure
        plt.savefig("./output-data/GWR-results/PNG/GWR_R2_" + dep + ".png")
        #plt.show()

        # create a gdf with the local R2 values
        Regression_21_24_R2 = Regression_21_24[["LSOA21CD",
                                                "LSOA21NM","s-ASG_ABC1",
                                                # "s-ASG_C2DE",
                                                "s-D3+",
                                                # "s-HHcars01",
                                                "s-HHcars2+",
                                                "s-HHT_own",
                                                # "s-HHT_rent",
                                                "s-semi-det",
                                                "s-flat",
                                                "s-terr-oth",
                                                # "Med_HP_21",
                                                "Med_HP_23",
                                                "Pop_dens",
                                                #"RoadKmDen",
                                                "POI_dens",
                                                #"job_th_21",
                                                #"job_th_23",
                                                "geometry",
                                                "x",
                                                "y",
                                                "localR2"]]

        # rename the columns with names shorter than 10 characters
        '''Regression_21_24_R2.rename(columns={"s-HHcars_2+": "s-HHcars2+",
                                            "s-Acc_det-semidet": "det_semidt",
                                            "Med_HP_2023": "Med_HP_23"},
                                 inplace=True)
        '''

        # Save the results to a shp file
        Regression_21_24_R2.to_file("./output-data/GWR-results/SHP/GWR_R2_" + dep + "_" + ".shp")

        # copy the independent variables into a new list
        labels = cat_indep_variables.copy()
        # insert the intercept as the first label
        labels.insert(0, 'Intercept')

        # Map local coefficients, only map the ones that are statistically significant based on the t-values
        for param in range(1, results.params.shape[1]):
            # Mask for statistically significant areas
            significant_mask = np.abs(results.tvalues[:, param]) > 1.96

            # Check if there are any significant areas to plot
            if significant_mask.any():
                fig, ax = plt.subplots(1,
                                       figsize=(8, 8),
                                       dpi=100,
                                       subplot_kw=dict(aspect='equal'))

                # add local coefficients to df using independent variable name:
                Regression_21_24[str(param)] = results.params[:, param]

                # Compute value ranges only for significant areas
                significant_values = Regression_21_24.loc[significant_mask, str(param)]
                vmin, vmax = significant_values.min(), significant_values.max()

                max_abs_val = max(abs(vmin), abs(vmax))

                # Define the colormap and normalization
                if vmin < 0 and vmax > 0:
                    # Diverging colormap for both positive and negative values
                    cmap = 'RdYlGn'
                    norm = mcolors.TwoSlopeNorm(vmin=min(vmin, -max_abs_val),
                                                vcenter=0,
                                                vmax=max(vmax, max_abs_val))
                elif vmax <= 0:
                    # Only negative values, use a red gradient
                    cmap = 'Reds_r'
                    norm = plt.Normalize(vmin=vmin, vmax=0)
                elif vmin >= 0:
                    # Only positive values, use a blue gradient
                    cmap = 'YlGn'
                    norm = plt.Normalize(vmin=0, vmax=vmax)

                # Plot only the significant areas
                Regression_21_24[significant_mask].plot(
                    str(param),
                    markersize=10.,
                    edgecolor='#555555',
                    linewidths=.25,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    norm=norm,
                    ax=ax,
                    zorder=2
                )

                # impose LSOA boundaries
                lsoa_boundaries(ax)

                ax.set_title(labels[param] + ' Coefficient estimates. Dep. var: ' + dep)
                fig = ax.get_figure()

                # Add colorbar
                sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                sm._A = []
                # fig.colorbar(sm)
                cbar = fig.colorbar(sm, ax=ax)  # This ensures the colorbar is linked to the correct axis

                _ = ax.axis('off')

                # Add tvales as a column to the dataframe
                Regression_21_24['tvalues_' + str(param)] = results.tvalues[:, param]

                # Save the figure
                plt.savefig("./output-data/GWR-results/PNG/GWR_coeff_" + dep + "_" + labels[param] + ".png")
                #plt.show()

                # Save the results to a shp file masking the significant areas
                Regression_21_24[significant_mask].to_file("./output-data/GWR-results/SHP/GWR_results_" + dep + "_" + str(param) + "_" + labels[param] + ".shp")
                #Regression_21_24.to_file("./output-data/GWR-results/SHP/GWR_results_" + dep + ".shp")

########################################################################################################################
# Check the distribution of the dependent variables
check_distribution_flag = False
#variables_to_test = Regression_21_24[[#"accessibility_21_arcsinh",
                                      #"accessibility_24_arcsinh",
                                      #"acc_diff_24_21_arcsinh"]]
                                      #"EVCI_improvement",
                                      #"EV_licensing_improvement"]]
                                      #"EVCI_improvement_rate",
                                      #"EV_licensing_improvement_rate",
                                      #"EVCI_improvement_rate_arcsinh",
                                      #"EV_licensing_improvement_rate_arcsinh"]]

#plot probability distribution function

# plot the distribution of the dependent variables without using sns
if check_distribution_flag == True:
    for var in variables_to_test:
        plt.figure(figsize=(8, 6))
        plt.hist(Regression_21_24[var], bins=30, color='blue', alpha=0.7)
        plt.title(f"Distribution of {var}")
        plt.xlabel(var)
        plt.ylabel("Frequency")
        plt.grid()
        plt.show()

# Check normality of the dependent variables
check_normality_flag = False
if check_normality_flag == True:
    from scipy import stats
    for var in variables_to_test:
        # Shapiro-Wilk test for normality
        stat, p = stats.shapiro(Regression_21_24[var])
        print("--------------------------------------------")
        print(f"Shapiro-Wilk test for normality for {var}:")
        print(f"Statistics={stat:.3f}, p={p:.3f}")
        # Interpret
        alpha = 0.05
        if p > alpha:
            print(f"{var} sample looks Gaussian (fail to reject H0)")
        else:
            print(f"{var} sample does not look Gaussian (reject H0)")

########################################################################################################################
# Correlation matrix
CSCA_2021_2024_flag = True

if CSCA_2021_2024_flag == True:
    # create a df for the correlation matrix
    Regression_21_24.rename(columns={"acc_21": "Aj_t1",
                                     "acc_24": "Aj_t2",
                                     "acc_diff": "Aj_diff",
                                     "s_impr": "SVj",
                                     "EVCI_improvement_rate": "s_imp_rate",
                                     "d_impr": "DVj",
                                     "EV_licensing_improvement_rate": "d_imp_rate",
                                     "s-HHcars_01": "s-HHcars01",
                                     "s-HHcars_2+": "s-HHcars2+",
                                     "s-HHT_owned": "s-HHT_own",
                                     "s-HHT_rented": "s-HHT_rent",
                                     "s-Acc_det-semidet": "s-semi-det",
                                     "s-Acc_flat": "s-flat",
                                     "s-Acc_other": "s-terr-oth",
                                     "Med_HP_2021": "Med_HP_21",
                                     "Med_HP_2023": "Med_HP_23",
                                     "Pop_density": "Pop_dens"}, inplace=True)
    print(Regression_21_24.columns)
    corr_matrix_2021_2024 = Regression_21_24[["Aj_t1", "Aj_t2", "Aj_diff",
                                              "SVj",
                                              #"s_imp_rate",
                                              "DVj",
                                              #"d_imp_rate",
                                              "s-ASG_ABC1", "s-ASG_C2DE",
                                              "s-D3+",
                                              "s-HHcars01", "s-HHcars2+",
                                              "s-HHT_own", "s-HHT_rent",
                                              "s-semi-det", "s-flat", "s-terr-oth",
                                              "Med_HP_21", "Med_HP_23",
                                              "Pop_dens",
                                              #"RoadKmDen",
                                              "POI_dens"]]
                                              #"job_th_21", "job_th_23"]]

    plt_and_save_corr_matrix(corr_matrix_2021_2024, outputs["correlation_matrix_2021_2024"])

########################################################################################################################
# Calculate the Gini coefficient for accessibility in 2021 and 2024
calculate_Gini_flag = False

if calculate_Gini_flag == True:
    # Load the accessibility data, make them numpy arrays
    accessibility_21 = np.array(Regression_21_24["acc_21"])
    accessibility_24 = np.array(Regression_21_24["acc_24"])

    # Remove NaNs
    accessibility_21 = accessibility_21[~np.isnan(accessibility_21)]
    accessibility_24 = accessibility_24[~np.isnan(accessibility_24)]

    # check if the data contains negative values
    if (accessibility_21 < 0).any() or (accessibility_24 < 0).any():
        raise ValueError("Data contains negative values. Gini coefficient cannot be calculated.")

    def gini_index_and_plot(values, year):
        """
        Calculate the Gini index for a 1D array of values and plot the Lorenz curve.

        Parameters:
            values (array-like): The values to calculate the Gini index for.
            year (str): The year or label for the plot title.

        Returns:
            float: The Gini index, or None if it cannot be calculated.
        """
        # Ensure the values are a NumPy array
        values = np.array(values)

        # Handle cases where all values are zero or identical
        if len(values) == 0 or np.all(values == 0):
            print(f"No variability in values for {year}. Gini cannot be calculated.")
            return None

        # Sort values in ascending order
        sorted_values = np.sort(values)

        # Handle cases where all values are identical
        if np.all(sorted_values == sorted_values[0]):
            print(f"All values are identical for {year}. Gini = 0 (perfect equality).")
            return 0.0

        # Calculate the cumulative sum of the values
        cumulative_values = np.cumsum(sorted_values)
        total = cumulative_values[-1]

        # Calculate the cumulative population and value proportions
        n = len(values)
        cumulative_population = np.linspace(0, 1, n + 1)  # Include 0 at the start for proper Lorenz curve
        cumulative_share = np.concatenate([[0], cumulative_values / total])  # Include 0 at the start for Lorenz curve

        # Use the Gini formula
        gini = 1 - 2 * np.sum((cumulative_share[:-1] + cumulative_share[1:]) * np.diff(cumulative_population))

        # Plot the Lorenz curve
        plt.figure(figsize=(8, 8))
        plt.plot(cumulative_population, cumulative_share, label="Lorenz Curve", color="blue", linewidth=2)
        plt.plot([0, 1], [0, 1], label="Line of Equality", color="red", linestyle="--", linewidth=1.5)  # Equality line
        plt.fill_between(cumulative_population, cumulative_share, cumulative_population,
                         color="blue", alpha=0.2, label=f"Gini: {gini:.4f}")
        plt.title(f"Lorenz Curve and Gini Coefficient for {year}")
        plt.xlabel("Cumulative Population Proportion")
        plt.ylabel("Cumulative Value Proportion")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()

        return gini

    # Calculate and plot Gini index for both years
    gini_2021 = gini_index_and_plot(accessibility_21, "2021")
    gini_2024 = gini_index_and_plot(accessibility_24, "2024")

    print(f"Gini Index for 2021: {gini_2021}")
    print(f"Gini Index for 2024: {gini_2024}")

########################################################################################################################
# Calculate summary statistics for the variables used in the regression from the Regression_21_24 data frame
calculate_summary_statistics_flag = False

if calculate_summary_statistics_flag == True:
    # Create a DataFrame with the variables of interest
    df_sm = Regression_21_24[[ 'acc_21', 'acc_24', 'acc_diff', 's_impr',
       's_imp_rate', 'd_impr', 'd_imp_rate', 's-ASG_ABC1', 's-ASG_C2DE',
       's-D3+', 's-HHcars01', 's-HHcars2+', 's-HHT_own', 's-HHT_rent',
       's-semi-det', 's-flat', 's-terr-oth', 'Med_HP_21', 'Med_HP_23',
       'Pop_dens', 'POI_dens']]

    # Calculate summary statistics
    summary_stats_df = df_sm.describe()

    # Save the summary statistics to a CSV file
    summary_stats_df.to_csv(outputs["summary_statistics"], index=False)
