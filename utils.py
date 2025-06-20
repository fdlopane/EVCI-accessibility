'''
Collection of functions used in the EVCI accessibility analysis
'''
from config import *

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm
import osmnx as ox
from shapely.validation import make_valid
from shapely.geometry import Polygon, Point
import overpy
from libpysal.weights import KNN

from esda.moran import Moran

from libpysal.weights import Queen
from spreg import OLS
from spreg import ML_Lag

from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap



########################################################################################################################
# correlation between 2 fields

def calculate_corr_matrix_2_var(analysis_df, field_1, field_2):
    formula = field_1 + ' ~ ' + field_2
    model_supply_demand = sm.formula.ols(formula, analysis_df).fit()
    model_supply_demand.summary()

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
    # ax.set_xlim(0, 1000)
    # ax.set_ylim(0, 40)

    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)

    # Show the legend
    ax.legend()

    # plt.savefig('Plot_final_0826/Walk_and_Transit_Level_4.png', format='png', dpi=120)

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
    # plt.savefig('Plot/OLS_regression_results_Level_4_WT.png', bbox_inches='tight', dpi=300)

    # Display the image
    plt.show()

########################################################################################################################
# correlation matrix

def plt_and_save_corr_matrix(analysis_df, output_file_name):
    # Pearson correlation

    # Plot the correlation matrix
    plt.rcParams["axes.grid"] = False
    f = plt.figure(figsize=(10, 10))

    # Plot the correlation matrix with a 'coolwarm' color map
    corr_matrix = analysis_df.corr(method='pearson')

    N = len(corr_matrix.columns)  # Number of columns
    M = len(corr_matrix.index)  # Number of rows

    ylabels = corr_matrix.columns.tolist()
    xlabels = corr_matrix.index.tolist()

    # Create mesh grid
    x, y = np.meshgrid(np.arange(M), np.arange(N))

    # Get the correlation values as a NumPy array
    corr_values = corr_matrix.values

    # Use absolute correlation values for size and original values for color
    # Avoid zero radius: add small epsilon if needed
    abs_corr = np.abs(corr_values)
    R = abs_corr / abs_corr.max() / 2  # Normalize radii

    # Create circle patches
    #circles = [plt.Circle((j, i), radius=(r * abs(1.5 - r))) for r, j, i in zip(R.flat, x.flat, y.flat)]
    circles = [plt.Circle((j, i), radius=(0.5*(0.3+3.14*r*r))) for r, j, i in zip(R.flat, x.flat, y.flat)]

    # Define custom colormap
    BluGrn = LinearSegmentedColormap.from_list("BluGrn",
                                               ["#C4E6C3FF", "#96D2A4FF", "#6DBC90FF", "#4DA284FF", "#36877AFF",
                                                "#266B6EFF", "#1D4F60FF"])
    Earth = LinearSegmentedColormap.from_list("Earth", ["#A16928FF", "#BD925AFF", "#D6BD8DFF", "#EDEAC2FF", "#B5C8B8FF",
                                                        "#79A7ACFF", "#2887A1FF"])
    BluRd = LinearSegmentedColormap.from_list("BluRd", ["#ca0020", "#f4a582", "#f7f7f7", "#92c5de", "#0571b0"])
    BluYlRd = LinearSegmentedColormap.from_list("BluYlRd", ["#d7191c", "#fdae61", "#ffffbf", "#abd9e9", "#2c7bb6"])
    Fall = LinearSegmentedColormap.from_list("Fall", ["#3D5941FF", "#778868FF", "#B5B991FF", "#F6EDBDFF", "#EDBB8AFF", "#DE8A5AFF", "#CA562CFF"])
    InvFall = LinearSegmentedColormap.from_list("InvFall", ["#CA562CFF", "#DE8A5AFF", "#EDBB8AFF", "#F6EDBDFF", "#B5B991FF", "#778868FF", "#3D5941FF"])
    InvFall = LinearSegmentedColormap.from_list("InvFall", ["#CA562CFF", "#F6EDBDFF", "#3D5941FF"])

    # Use correlation values directly for coloring
    col = PatchCollection(circles, array=corr_values.flatten(), cmap=InvFall, linewidth=0.5)  # copper
    fig, ax = plt.subplots()
    ax.add_collection(col)

    # Set axis labels and ticks
    ax.set(xticks=np.arange(M), yticks=np.arange(N),
           xticklabels=xlabels, yticklabels=ylabels)
    ax.set_xticks(np.arange(M + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(N + 1) - 0.5, minor=True)
    # ax.grid(which='minor')

    fig.colorbar(col)
    plt.xticks(rotation=90)
    plt.tight_layout()
    #plt.show()

    # Now save the figure as a png file
    plt.savefig(outputs["correlation_matrix_2021_2024_figure"], format='png', bbox_inches='tight', dpi=300)


    '''
    #plt.matshow(corr_matrix, fignum=f.number, cmap='RdBu_r')
    
    plt.matshow(corr_matrix, cmap='RdBu_r')

    # Add a color bar using the same color map as the heatmap
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)

    # Add tick labels
    plt.xticks(range(analysis_df.shape[1]), analysis_df.columns, fontsize=10, rotation=90)
    plt.yticks(range(analysis_df.shape[1]), analysis_df.columns, fontsize=10)

    # Add the correlation coefficient values to each cell
    for (i, j), val in np.ndenumerate(corr_matrix.values):
        plt.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10, color='black')

    # Add a title
    #plt.title('Correlation Matrix', fontsize=10)

    # # Save the image as a PNG file
    # plt.savefig('Plot/correlation_matrix_improved.png', format='png', dpi=300, bbox_inches='tight')

    # Display the chart
    #plt.show()

    # print(corr_matrix)
    # Save correlation matrix to csv
    corr_matrix.to_csv(output_file_name)

    return corr_matrix
    '''

########################################################################################################################
# OLS Analysis

def OLS_analysis(analysis_df, dependent_variable, independent_variables):
    # Remove NaNs
    analysis_2021 = analysis_df.dropna()

    print()
    print("###########################################################################################################")
    print("OLS analysis for: ", dependent_variable)
    # Define the list of variables:
    models = []
    for i in independent_variables:
        #print("Independent variable: ", i)
        X = analysis_2021[i]
        Y = analysis_2021[dependent_variable]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X.astype(float)).fit()
        models.append(model)
        #print(model.summary())

    # Create a summary table for each model in models
    summary_table = pd.DataFrame(columns=['Model', 'Variable', 'Coefficient', 'Standard Error', 't-Value', 'p-Value',
                                          'CI 2.5%', 'CI 97.5%'])

    # Collect all rows in a list
    rows_to_add = []

    # Iterate through models and their independent variables
    for i, model in enumerate(models):
        for var in model.params.index:  # Get variable names
            if var == 'const':  # Skip the constant term if present
                continue

            # Skip variables not found in pvalues or conf_int
            if var not in model.pvalues or var not in model.conf_int().index:
                print(f"Warning: Variable '{var}' not found in model.pvalues or model.conf_int()")
                continue

            # Extract confidence intervals
            ci_lower, ci_upper = model.conf_int().loc[var]

            # Create a row dictionary
            row = {
                'Model': i + 1,
                'Variable': var,
                'Coefficient': model.params[var],
                'Standard Error': model.bse[var],
                't-Value': model.tvalues[var],
                'p-Value': model.pvalues[var],
                'CI 2.5%': ci_lower,
                'CI 97.5%': ci_upper
            }
            rows_to_add.append(row)

    # Convert the list of rows into a DataFrame and concatenate with the summary table
    summary_table = pd.concat([summary_table, pd.DataFrame(rows_to_add)], ignore_index=True)

    # Sort the summary table by Model and Variable
    summary_table.sort_values(by=['Model', 'Variable'], inplace=True)

    # Add a column for significance
    summary_table['Significance'] = ''
    summary_table.loc[summary_table['p-Value'] <= 0.01, 'Significance'] = '***'
    summary_table.loc[(summary_table['p-Value'] > 0.01) & (summary_table['p-Value'] <= 0.05), 'Significance'] = '**'
    summary_table.loc[(summary_table['p-Value'] > 0.05) & (summary_table['p-Value'] <= 0.1), 'Significance'] = '*'
    # Move the significance column straight after the p-Value column
    cols = summary_table.columns.tolist()
    cols = cols[:6] + cols[-1:] + cols[6:-1]
    summary_table = summary_table[cols]

    return summary_table

########################################################################################################################
def OLS_analysis_univariate_moran(analysis_df, dependent_variable, independent_variables, x_col='x', y_col='y', k=8):
    # Remove NaNs
    analysis_2021 = analysis_df.dropna()

    print()
    print("###########################################################################################################")
    print("OLS analysis for: ", dependent_variable)

    # Define the list of variables:
    models = []
    residuals_list = []

    for i in independent_variables:
        # Get data for the univariate model
        X = analysis_2021[i]
        Y = analysis_2021[dependent_variable]
        X = sm.add_constant(X)

        # Fit the model
        model = sm.OLS(Y, X.astype(float)).fit()
        models.append(model)

        # Store residuals for Moran's I calculation
        residuals_list.append(model.resid)

    # Create a summary table for each model in models
    summary_table = pd.DataFrame(columns=['Model', 'Variable', 'Coefficient', 'Standard Error', 't-Value', 'p-Value',
                                          'CI 2.5%', 'CI 97.5%', 'R2'])

    # Collect all rows in a list
    rows_to_add = []

    # Iterate through models and their independent variables
    for i, model in enumerate(models):
        for var in model.params.index:  # Get variable names
            if var == 'const':  # Skip the constant term if present
                continue

            # Skip variables not found in pvalues or conf_int
            if var not in model.pvalues or var not in model.conf_int().index:
                print(f"Warning: Variable '{var}' not found in model.pvalues or model.conf_int()")
                continue

            # Extract confidence intervals
            ci_lower, ci_upper = model.conf_int().loc[var]

            # Create a row dictionary with R2 added (model.rsquared)
            row = {
                'Model': i + 1,
                'Variable': var,
                'Coefficient': model.params[var],
                'Standard Error': model.bse[var],
                't-Value': model.tvalues[var],
                'p-Value': model.pvalues[var],
                'CI 2.5%': ci_lower,
                'CI 97.5%': ci_upper,
                'R2': model.rsquared
            }
            rows_to_add.append(row)

    # Convert the list of rows into a DataFrame and concatenate with the summary table
    summary_table = pd.concat([summary_table, pd.DataFrame(rows_to_add)], ignore_index=True)

    # Sort the summary table by Model and Variable
    summary_table.sort_values(by=['Model', 'Variable'], inplace=True)

    # Add a column for significance based on p-value thresholds
    summary_table['Significance'] = ''
    summary_table.loc[summary_table['p-Value'] <= 0.01, 'Significance'] = '***'
    summary_table.loc[(summary_table['p-Value'] > 0.01) & (summary_table['p-Value'] <= 0.05), 'Significance'] = '**'
    summary_table.loc[(summary_table['p-Value'] > 0.05) & (summary_table['p-Value'] <= 0.1), 'Significance'] = '*'

    # Reorder columns so that 'Significance' comes straight after 'p-Value'
    ordered_cols = ['Model', 'Variable', 'Coefficient', 'Standard Error', 't-Value', 'p-Value', 'Significance', 'R2', 'CI 2.5%', 'CI 97.5%']
    summary_table = summary_table[ordered_cols]

    # Use residuals from the last model (or combine appropriately if needed)
    residuals = model.resid.values

    # Spatial weights matrix using k-nearest neighbors (or based on a threshold distance)
    from libpysal.weights import KNN
    coords = list(zip(analysis_2021[x_col], analysis_2021[y_col]))
    w = KNN.from_array(coords, k=k)

    # Calculate Moran's I
    from esda.moran import Moran
    moran = Moran(residuals, w)

    print(f"\nMoran’s I: {moran.I:.4f}")
    print(f"p-value: {moran.p_sim:.4f} (based on {moran.permutations} permutations)")

    return summary_table, residuals, moran

########################################################################################################################
def OLS_analysis_univariate(analysis_df, dependent_variable, independent_variables):
    # Remove NaNs
    analysis_2021 = analysis_df.dropna()

    print()
    print("###########################################################################################################")
    print("OLS analysis for:", dependent_variable)

    # Prepare the data for multivariate OLS
    X = analysis_2021[independent_variables]
    Y = analysis_2021[dependent_variable]
    X = sm.add_constant(X)  # Add intercept
    model = sm.OLS(Y, X.astype(float)).fit()

    # Create a summary table
    summary_table = pd.DataFrame(columns=['Variable', 'Coefficient', 'Standard Error', 't-Value', 'p-Value',
                                          'CI 2.5%', 'CI 97.5%'])

    rows_to_add = []

    for var in model.params.index:
        if var not in model.pvalues or var not in model.conf_int().index:
            print(f"Warning: Variable '{var}' not found in model.pvalues or model.conf_int()")
            continue

        ci_lower, ci_upper = model.conf_int().loc[var]

        row = {
            'Variable': var,
            'Coefficient': model.params[var],
            'Standard Error': model.bse[var],
            't-Value': model.tvalues[var],
            'p-Value': model.pvalues[var],
            'CI 2.5%': ci_lower,
            'CI 97.5%': ci_upper
        }
        rows_to_add.append(row)

    summary_table = pd.concat([summary_table, pd.DataFrame(rows_to_add)], ignore_index=True)

    # Add significance column
    summary_table['Significance'] = ''
    summary_table.loc[summary_table['p-Value'] <= 0.01, 'Significance'] = '***'
    summary_table.loc[(summary_table['p-Value'] > 0.01) & (summary_table['p-Value'] <= 0.05), 'Significance'] = '**'
    summary_table.loc[(summary_table['p-Value'] > 0.05) & (summary_table['p-Value'] <= 0.1), 'Significance'] = '*'

    # Reorder columns
    cols = summary_table.columns.tolist()
    cols = cols[:5] + ['Significance'] + cols[5:-1]
    summary_table = summary_table[cols]

    return summary_table

########################################################################################################################

def OLS_analysis_multivariate_moran(analysis_df, dependent_variable, independent_variables, x_col='x', y_col='y', k=8):
    # Remove NaNs using the required columns
    analysis_2021 = analysis_df.dropna(subset=independent_variables + [dependent_variable, x_col, y_col])

    print()
    print("###########################################################################################################")
    print("OLS analysis for:", dependent_variable)

    # Prepare the data for multivariate OLS
    X = analysis_2021[independent_variables]
    Y = analysis_2021[dependent_variable]
    X = sm.add_constant(X)
    model = sm.OLS(Y, X.astype(float)).fit()

    # Create a summary table with an added R2 column
    summary_table = pd.DataFrame(columns=['Variable', 'Coefficient', 'Standard Error', 't-Value', 'p-Value',
                                            'CI 2.5%', 'CI 97.5%', 'R2'])

    rows_to_add = []

    # Iterate through each parameter in the model
    for var in model.params.index:
        if var not in model.pvalues or var not in model.conf_int().index:
            print(f"Warning: Variable '{var}' not found in model.pvalues or model.conf_int()")
            continue

        ci_lower, ci_upper = model.conf_int().loc[var]

        row = {
            'Variable': var,
            'Coefficient': model.params[var],
            'Standard Error': model.bse[var],
            't-Value': model.tvalues[var],
            'p-Value': model.pvalues[var],
            'CI 2.5%': ci_lower,
            'CI 97.5%': ci_upper,
            'R2': model.rsquared
        }
        rows_to_add.append(row)

    summary_table = pd.concat([summary_table, pd.DataFrame(rows_to_add)], ignore_index=True)

    # Add significance column based on p-values
    summary_table['Significance'] = ''
    summary_table.loc[summary_table['p-Value'] <= 0.01, 'Significance'] = '***'
    summary_table.loc[(summary_table['p-Value'] > 0.01) & (summary_table['p-Value'] <= 0.05), 'Significance'] = '**'
    summary_table.loc[(summary_table['p-Value'] > 0.05) & (summary_table['p-Value'] <= 0.1), 'Significance'] = '*'

    # Reorder columns so that the Significance column comes right after the p-Value column
    cols = summary_table.columns.tolist()
    # This reordering: keep first 5 columns, then 'Significance', then remaining columns.
    cols = cols[:5] + ['Significance'] + cols[5:-1] + [cols[-1]]
    summary_table = summary_table[cols]

    # Residuals from the fitted model
    residuals = model.resid.values

    # Create spatial weights matrix using k-nearest neighbors; you can adjust k as needed
    from libpysal.weights import KNN
    coords = list(zip(analysis_2021[x_col], analysis_2021[y_col]))
    w = KNN.from_array(coords, k=k)

    # Calculate Moran's I for the residuals
    from esda.moran import Moran
    moran = Moran(residuals, w)

    print(f"\nMoran’s I: {moran.I:.4f}")
    print(f"p-value: {moran.p_sim:.4f} (based on {moran.permutations} permutations)")

    return summary_table, residuals, moran

########################################################################################################################
# Road network density calculation
def road_net_chars_calculator(bfcs):
    '''
    This function calculates the road network characteristics for each LSOA in the input bfcs dataframe
    '''
    bfcs['geometry_4326'] = bfcs.geometry.to_crs(4326)

    ## This will output a pandas dataframe of street network characteristics

    chars_dict = {lsoa_cd: {} for lsoa_cd in bfcs.LSOA21CD}

    for idx, row in tqdm(bfcs.iterrows()):
        if (len(chars_dict[row.LSOA21CD]) == 0) | (chars_dict[row.LSOA21CD] == 'error'):
            poly_4326 = row.geometry_4326
            if not poly_4326.is_valid:
                poly_4326 = make_valid(poly_4326)
            poly_27700 = row.geometry
            if not poly_27700.is_valid:
                poly_27700 = make_valid(poly_27700)

            try:
                G = ox.graph_from_polygon(poly_4326, network_type='all')  # 'drive', 'bike', 'walk', etc.
                stats = ox.stats.basic_stats(G)
                #chars_dict[row.LSOA21CD]['circuity'] = stats['circuity_avg']
                #chars_dict[row.LSOA21CD]['self_loops'] = stats['self_loop_proportion']
                chars_dict[row.LSOA21CD]['street_length_total'] = stats['street_length_total']
                chars_dict[row.LSOA21CD]['street_density_km'] = stats['street_length_total'] / poly_27700.area
                chars_dict[row.LSOA21CD]['intersection_count'] = stats['intersection_count']
                chars_dict[row.LSOA21CD]['intersection_density_km'] = stats['intersection_count'] / poly_27700.area
                #chars_dict[row.LSOA21CD]['streets_per_node'] = stats['streets_per_node_avg']

            except:
                chars_dict[row.LSOA21CD] = 'error'

        else:
            pass

    chars_df = pd.DataFrame(chars_dict).transpose().reset_index().rename(columns={'index': 'LSOA21CD'})

    return chars_df

########################################################################################################################
# Amenities scraping
def amenities_scarping(bfcs):
    '''
    This function scrapes the amenities data from OpenStreetMap
    '''

    bfcs['geometry_4326'] = bfcs.geometry.to_crs(4326)

    ## This will output a geopandas dataframe of amenities from OSM and their associated LSOA

    xmin, ymin, xmax, ymax = bfcs.to_crs(4326).total_bounds
    xmin -= .25
    ymin -= .25
    ymax += .25
    xmax += .25

    xs = np.linspace(xmin, xmax, 10)
    ys = np.linspace(ymin, ymax, 10)
    polys = []
    for i in range(9):
        for j in range(9):
            poly = Polygon([(xs[i], ys[j]), (xs[i + 1], ys[j]), (xs[i + 1], ys[j + 1]), (xs[i], ys[j + 1])])
            polys.append(poly)
    grid = gpd.GeoDataFrame({'geometry': polys, 'grid_id': range(len(polys))}, crs=4326)
    grid = gpd.sjoin(grid, bfcs.to_crs(4326))
    grid = grid.drop_duplicates('grid_id')[['geometry']]

    api = overpy.Overpass()

    results_dfs = []
    for g in tqdm(grid.geometry):
        bbox = g.bounds
        query_bodies = [f'node["amenity"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});',
                        f'node["shop"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});',
                        f'node["bus"~"station|stop"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});',
                        f'node["railway"~"station|stop"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});']
        for q in query_bodies:
            query = f"""
                [out:json];
                (
                {q}
                );
                out body;
                """

            result = api.query(query)
            amenities_data = []

            for node in result.nodes:
                amenities_data.append({
                    "id": node.id,
                    "name": node.tags.get("name", "N/A"),
                    "type": node.tags.get("amenity", "N/A"),
                    "bus": node.tags.get("bus", "N/A"),
                    "railway": node.tags.get("railway", "N/A"),
                    "latitude": node.lat,
                    "longitude": node.lon
                })

            results_dfs.append(pd.DataFrame(amenities_data))

    amenities = pd.concat(results_dfs)

    amenities = amenities.drop_duplicates(subset=['id'])
    amenities['geometry'] = amenities.apply(lambda row: Point(row.longitude, row.latitude), axis=1)

    amenities = gpd.GeoDataFrame(amenities, crs=4326)
    amenities = gpd.sjoin(amenities, bfcs.to_crs(4326)[['LSOA21CD', 'geometry']])
    # amenities.drop(['latitude','longitude'],axis=1).to_file(data_path + 'amenities.geojson',driver='GeoJSON')

    return amenities

########################################################################################################################
# Spatial lag model (SLM)
def run_slm(df, dep_var, indep_vars, w=None):
    """
    Runs a Spatial Lag Model (SLM) using ML_Lag and returns a tidy DataFrame
    with coefficients, p-values, significance stars, and model diagnostics,
    including the R2 (pseudo R-squared) value.

    Each row represents an estimated parameter (intercept or coefficient). The
    significance column is inserted immediately after the p_value column.
    """
    # Create spatial weights if not provided
    if w is None:
        w = Queen.from_dataframe(df, use_index=False)
    w.transform = 'r'

    # Prepare data
    y = df[[dep_var]].values
    X = df[indep_vars].values

    # Fit model using ML_Lag
    model = ML_Lag(y, X, w=w, name_y=dep_var, name_x=indep_vars)

    # Define coefficient names: constant + independent variable names
    coef_names = ['constant'] + indep_vars
    # z_stat returns a list of tuples: (z_value, p_value) for each coefficient,
    # and the last entry is for rho.
    z_stats = model.z_stat

    # Prepare rows for the output DataFrame (using model.pr2 for R2)
    rows = []
    for name, (coef, (z, p)) in zip(coef_names, zip(model.betas.flatten(), z_stats[:-1])):
        # Assign significance based on conventional thresholds
        if p <= 0.01:
            sig = '***'
        elif p <= 0.05:
            sig = '**'
        elif p <= 0.1:
            sig = '*'
        else:
            sig = ''

        rows.append({
            'dependent_var': dep_var,
            'variable': name,
            'coefficient': coef,
            'p_value': p,
            'R2': model.pr2,  # Model's pseudo R-squared added here
            'significance': sig,
            'rho': model.rho,  # model.rho is a scalar
            'rho_p_value': z_stats[-1][1],  # Last z_stat corresponds to rho
            'log_likelihood': model.logll,
            'AIC': model.aic,
            'BIC': model.schwarz,
        })

    # Create DataFrame and order columns so that significance is after p_value and R2 appears next
    df_results = pd.DataFrame(rows)
    ordered_cols = ['dependent_var', 'variable', 'coefficient', 'p_value', 'R2', 'significance',
                    'rho', 'rho_p_value', 'log_likelihood', 'AIC', 'BIC']
    df_results = df_results[ordered_cols]
    return df_results

########################################################################################################################
# calculate the LM-lag, LM-error, Robust LM-lag, and Robust LM-error tests

def lm_tests_multiple_models(df, dep_vars, indep_vars_list):
    """
    Perform spatial LM diagnostics (LM-lag, LM-error, Robust LM-lag/error)
    for multiple models, each with its own set of independent variables.

    Parameters:
    - df: pandas DataFrame with all relevant variables
    - dep_vars: list of dependent variable names (one per model)
    - indep_vars_list: list of lists of independent variable names, matching dep_vars

    Returns:
    - pandas DataFrame summarizing diagnostics
    """
    if len(dep_vars) != len(indep_vars_list):
        raise ValueError("Length of dep_vars and indep_vars_list must match.")

    # Create spatial weights
    w = Queen.from_dataframe(df)
    w.transform = 'r'

    results = []

    for y_var, x_vars in zip(dep_vars, indep_vars_list):
        try:
            y = df[[y_var]].values
            X = df[x_vars].values

            model = OLS(y, X, w=w, name_y=y_var, name_x=x_vars, nonspat_diag=True, spat_diag=True)

            # Diagnostics
            lm_lag, lm_lag_p = getattr(model, 'lm_lag', (np.nan, np.nan))
            lm_error, lm_error_p = getattr(model, 'lm_error', (np.nan, np.nan))
            rlm_lag, rlm_lag_p = getattr(model, 'rlm_lag', (np.nan, np.nan))
            rlm_error, rlm_error_p = getattr(model, 'rlm_error', (np.nan, np.nan))

            results.append({
                "Dependent Variable": y_var,
                "Independent Variables": ", ".join(x_vars),
                "LM Lag": lm_lag,
                "LM Lag p-value": lm_lag_p,
                "LM Error": lm_error,
                "LM Error p-value": lm_error_p,
                "Robust LM Lag": rlm_lag,
                "Robust LM Lag p-value": rlm_lag_p,
                "Robust LM Error": rlm_error,
                "Robust LM Error p-value": rlm_error_p,
            })

        except Exception as e:
            print(f"[ERROR] Model for {y_var} failed: {e}")
            results.append({
                "Dependent Variable": y_var,
                "Independent Variables": ", ".join(x_vars),
                "LM Lag": np.nan,
                "LM Lag p-value": np.nan,
                "LM Error": np.nan,
                "LM Error p-value": np.nan,
                "Robust LM Lag": np.nan,
                "Robust LM Lag p-value": np.nan,
                "Robust LM Error": np.nan,
                "Robust LM Error p-value": np.nan,
            })

    return pd.DataFrame(results)