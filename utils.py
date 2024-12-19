'''
Collection of functions used in the EVCI accessibility analysis
'''

import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    plt.matshow(corr_matrix, fignum=f.number, cmap='RdBu_r')

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
    plt.show()

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
    for i in independent_variables:
        X = analysis_2021[[i]]
        Y = analysis_2021[dependent_variable]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X.astype(float)).fit()
        #print(model.summary())

    # Save outputs of multiple OLS models into a single table
    # Create a list of models
    models = []
    for i in independent_variables:
        X = analysis_2021[[i]]
        Y = analysis_2021[dependent_variable]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        models.append(model)

    # Create a summary table including the coefficients, standard errors, t-values, p-values, and confidence intervals
    summary_table = pd.DataFrame(
        columns=['Variable', 'Coefficient', 'Standard Error', 't-Value', 'p-Value', 'CI 2.5%', 'CI 97.5%'])
    for i, model in enumerate(models):
        summary_table.loc[i] = [independent_variables[i], model.params[1], model.bse[1], model.tvalues[1],
                                model.pvalues[1], model.conf_int()[0][1], model.conf_int()[1][1]]

    print(summary_table)
    print("###########################################################################################################")
    print()
    return summary_table


########################################################################################################################
#