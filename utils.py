'''
Collection of functions used in the EVCI accessibility analysis
'''

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
    plt.title('Correlation Matrix', fontsize=10)

    # # Save the image as a PNG file
    # plt.savefig('Plot/correlation_matrix_improved.png', format='png', dpi=300, bbox_inches='tight')

    # Display the chart
    #plt.show()

    # print(corr_matrix)
    # Save correlation matrix to csv
    corr_matrix.to_csv(output_file_name)

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


    '''
    for i, model in enumerate(models):
        for j in independent_variables:
            # TODO: use concat to add the rows to the summary table
            summary_table = summary_table.append(
                {'Model': i + 1, 'Variable': j, 'Coefficient': model.params[j], 'Standard Error': model.bse[j],
                 't-Value': model.tvalues[j], 'p-Value': model.pvalues[j], 'CI 2.5%': model.conf_int()[0][j],
                 'CI 97.5%': model.conf_int()[1][j]}, ignore_index=True)
    '''

    '''
        summary_table.loc[i] = [i+1, independent_variables[i], model.params[1], model.bse[1], model.tvalues[1],
                                model.pvalues[1], '', model.conf_int()[0][1], model.conf_int()[1][1]]
    '''



    # Fill the significance column
    #summary_table.loc[summary_table['p-Value'] <= 0.01, 'Significance'] = '***'
    #summary_table.loc[(summary_table['p-Value'] > 0.01) & (summary_table['p-Value'] <= 0.05), 'Significance'] = '**'
    #summary_table.loc[(summary_table['p-Value'] > 0.05) & (summary_table['p-Value'] <= 0.1), 'Significance'] = '*'

    #print(summary_table)
    #print("###########################################################################################################")
    #print()
    return summary_table

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