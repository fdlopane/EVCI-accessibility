# The keys of the inputs and outputs dictionaries, as well as the file names should follow the camelCase notation.

inputs = {}
inputs["EVCI-LSOA"] = "./input-data/EVCI/EVCI-count-LSOA.csv"
inputs["EVCI-points-raw"] = "./input-data/EVCI/EVCI-points-raw.csv"
inputs["EV-counts-LSOA"] = "./input-data/EV_licensing_London/EV-licensing-LSOA-GLA.csv"
inputs["Mean_house_prices_GB"] = "./input-data/House_prices/Mean_price_residential_properties_LSOA_1995_2022_GB.csv"
inputs["Median_house_prices_GB"] = "./input-data/House_prices/Median_price_residential_properties_LSOA_1995_2022_GB.csv"
inputs["London_LSOAs"] = "./input-data/Geography/London_LSOAs.csv"
inputs["London_LSOA_centroids"] = "./input-data/Geography/London_LSOA_centroids.shp"
inputs["London_LSOA_polygons"] = "./input-data/Geography/London_LSOA_polygons.shp"
inputs["HH_deprivation_2021"] = "./input-data/Deprivation/HH_deprivation_London_LSOA_2021.csv"
inputs["ASG_GB"] = "./input-data/Approximate_social_grade/Approximated_social_grade_LSOA_2021_GB.csv"
inputs["Population_density_GB"] = "./input-data/Population/2021_LSOA_Population_GB.csv"
inputs["Vehicle_ownership_London_2021"] = "./input-data/Vehicle_ownership/Vehicles_HH_2021_London.csv"
inputs["House_tenure_London_2021"] = "./input-data/House_tenure/House_tenure_2021_London.csv"
inputs["Accommodation_type_London_2021"] = "./input-data/Accommodation_type/Accommodation_type_2021_London.csv"
inputs["Employment_2021_GB"] = "./input-data/Employment/employment_GB_2021.csv"
inputs["Employment_2023_GB"] = "./input-data/Employment/employment_GB_2023.csv"

generated = {}
generated["EVCI-London"] = "./generated-files/EVCI-London.csv"
generated["input-analytics-db"] = "./generated-files/input-supply-demand-cleanmerge.csv"
generated["input-totals"] = "./generated-files/input-supply-demand-totals.csv"
generated["Mean_house_prices_London"] = "./generated-files/Mean_price_residential_properties_LSOA_1995_2022_London.csv"
generated["Median_house_prices_London"] = "./generated-files/Median_price_residential_properties_LSOA_1995_2022_London.csv"
generated["ASG_London"] = "./generated-files/Approximated_social_grade_LSOA_2021_London.csv"
generated["road_net_chars"] = "./generated-files/road_network_characteristics.csv"
generated["POIs"] = "./generated-files/POIs.csv"

outputs = {}
outputs["correlation_matrix_2021_2024"] = "./output-data/correlation_matrix_2021_2024.csv"
outputs["OLS_supply_2021"] = "./output-data/OLS_supply_2021.csv"
outputs["OLS_demand_2021"] = "./output-data/OLS_demand_2021.csv"

# Accessibility 2021
outputs["OLS_accessibility_2021"] = "./output-data/OLS_accessibility_2021.csv"
outputs["OLS_accessibility_2021_residuals"] = "./output-data/OLS_accessibility_2021_residuals.csv"
outputs["OLS_accessibility_2021_moran"] = "./output-data/OLS_accessibility_2021_moran.csv"

outputs["univ_OLS_accessibility_2021"] = "./output-data/univ_OLS_accessibility_2021.csv"
outputs["univ_OLS_accessibility_2021_residuals"] = "./output-data/univ_OLS_accessibility_2021_residuals.csv"
outputs["univ_OLS_accessibility_2021_moran"] = "./output-data/univ_OLS_accessibility_2021_moran.csv"

# Accessibility 2024
outputs["OLS_accessibility_2024"] = "./output-data/OLS_accessibility_2024.csv"
outputs["OLS_accessibility_2024_residuals"] = "./output-data/OLS_accessibility_2024_residuals.csv"
outputs["OLS_accessibility_2024_moran"] = "./output-data/OLS_accessibility_2024_moran.csv"

outputs["univ_OLS_accessibility_2024"] = "./output-data/univ_OLS_accessibility_2024.csv"
outputs["univ_OLS_accessibility_2024_residuals"] = "./output-data/univ_OLS_accessibility_2024_residuals.csv"
outputs["univ_OLS_accessibility_2024_moran"] = "./output-data/univ_OLS_accessibility_2024_moran.csv"

# Accessibility difference 2021-2024
outputs["OLS_diff_accessibility_21_24"] = "./output-data/OLS_diff_accessibility_21_24.csv"
outputs["OLS_diff_accessibility_21_24_residuals"] = "./output-data/OLS_diff_accessibility_21_24_residuals.csv"
outputs["OLS_diff_accessibility_21_24_moran"] = "./output-data/OLS_diff_accessibility_21_24_moran.csv"

outputs["univ_OLS_diff_accessibility_21_24"] = "./output-data/univ_OLS_diff_accessibility_21_24.csv"
outputs["univ_OLS_diff_accessibility_21_24_residuals"] = "./output-data/univ_OLS_diff_accessibility_21_24_residuals.csv"
outputs["univ_OLS_diff_accessibility_21_24_moran"] = "./output-data/univ_OLS_diff_accessibility_21_24_moran.csv"

# Supply improvement
outputs["OLS_supply_improvement"] = "./output-data/OLS_supply_improvement.csv"
outputs["OLS_supply_improvement_residuals"] = "./output-data/OLS_supply_improvement_residuals.csv"
outputs["OLS_supply_improvement_moran"] = "./output-data/OLS_supply_improvement_moran.csv"

outputs["univ_OLS_supply_improvement"] = "./output-data/univ_OLS_supply_improvement.csv"
outputs["univ_OLS_supply_improvement_residuals"] = "./output-data/univ_OLS_supply_improvement_residuals.csv"
outputs["univ_OLS_supply_improvement_moran"] = "./output-data/univ_OLS_supply_improvement_moran.csv"

# Demand improvement
outputs["OLS_demand_improvement"] = "./output-data/OLS_demand_improvement.csv"
outputs["OLS_demand_improvement_residuals"] = "./output-data/OLS_demand_improvement_residuals.csv"
outputs["OLS_demand_improvement_moran"] = "./output-data/OLS_demand_improvement_moran.csv"

outputs["univ_OLS_demand_improvement"] = "./output-data/univ_OLS_demand_improvement.csv"
outputs["univ_OLS_demand_improvement_residuals"] = "./output-data/univ_OLS_demand_improvement_residuals.csv"
outputs["univ_OLS_demand_improvement_moran"] = "./output-data/univ_OLS_demand_improvement_moran.csv"

# Improvement rates
outputs["OLS_supply_improvement_rate"] = "./output-data/OLS_supply_improvement_rate.csv"
outputs["OLS_demand_improvement_rate"] = "./output-data/OLS_demand_improvement_rate.csv"


