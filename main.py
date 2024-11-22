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

# TODO: deal with null and -1 values is EV_counts
# EV_counts.fillna(0, inplace=True)

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

print(analysis_df.columns)

########################################################################################################################
# Remove EV licensing outliers (remove those LSOAs with licensing > lic_threshold)
lic_threshold = 1000
analysis_df = analysis_df.drop(analysis_df[analysis_df.y2024Q2 > lic_threshold].index)

########################################################################################################################

# Measure correlation between supply and demand
model_supply_demand = sm.formula.ols('EVCI2024 ~ y2024Q2', analysis_df).fit()
model_supply_demand.summary()

########################################################################################################################
# Scatter plot demand vs supply
# Get the regression model parameters and statistics
beta_0, beta_1 = model_supply_demand.params
rsq = model_supply_demand.rsquared
pval_0, pval_1 = model_supply_demand.pvalues

# Create a plot object
fig, ax = plt.subplots(figsize=(20, 10), dpi=120)

# Plot the scatter plot with custom colors
analysis_df.plot(kind='scatter',
                 x='y2024Q2',
                 y='EVCI2024',
                 ax=ax,
                 color='skyblue',
                 s=30,
                 edgecolor='black',
                 linewidths=0.6,
                 alpha=0.9)

# Plot the regression line
X = analysis_df.y2024Q2
ax.plot(X,
        X * beta_1 + beta_0,
        color='darkred',
        linewidth=2,
        linestyle='--',
        label=f'Regression Line\ny = {round(beta_1, 3)}x + {round(beta_0, 3)}')

# Set the title and axis labels
ax.set_title('EVCI Supply vs Demand', fontsize=16)
ax.set_xlabel('EV licensing counts', fontsize=14)
ax.set_ylabel('EVCI count', fontsize=14)

# Set the x and y axis limits
#ax.set_xlim(0, 1000)
#ax.set_ylim(0, 40)

# Add grid lines
ax.grid(True, which='both', linestyle='--', linewidth=0.7)

# Show the legend
ax.legend()

#plt.savefig('Plot_final_0826/Walk_and_Transit_Level_4.png', format='png', dpi=120)

# Display the chart
plt.show()

# Print the regression equation and other statistics
print()
print("----------------------------------------------------------")
print("Supply vs Demand regression equation and other statistics")
print("y =", round(beta_1, 3), "x +", round(beta_0, 3))
print("Rsq = ", rsq)
print("p-value = ", round(pval_1, 5))
print("----------------------------------------------------------")
print()
########################################################################################################################

# Plot regression statistics
# Convert the regression results to a string
result_summary_SD = model_supply_demand.summary().as_text()

# Create a Matplotlib figure
fig, ax = plt.subplots(figsize=(10, 6))  # You can adjust the size

# Hide the axes
ax.axis('off')

# Display the regression results text in the figure
ax.text(0, 1, result_summary_SD, fontsize=10, ha='left', va='top', family='monospace')

# Save as a PNG file
#plt.savefig('Plot/OLS_regression_results_Level_4_WT.png', bbox_inches='tight', dpi=300)

# Display the image
plt.show()
########################################################################################################################
