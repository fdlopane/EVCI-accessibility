# The keys of the inputs and outputs dictionaries, as well as the file names should follow the camelCase notation.

inputs = {}
inputs["EVCI-LSOA"] = "./input-data/EVCI/EVCI-count-LSOA.csv"
inputs["EVCI-points-raw"] = "./input-data/EVCI/EVCI-points-raw.csv"
inputs["EV-counts-LSOA"] = "./input-data/EV_licensing_London/EV-licensing-LSOA-GLA.csv"
inputs["Mean_house_prices_GB"] = "./input-data/House_prices/Mean_price_residential_properties_LSOA_1995_2022_GB.csv"
inputs["Median_house_prices_GB"] = "./input-data/House_prices/Median_price_residential_properties_LSOA_1995_2022_GB.csv"
inputs["London_LSOAs"] = "./input-data/Geography/London_LSOAs.csv"
inputs["HH_deprivation_2021"] = "./input-data/Deprivation/HH_deprivation_London_LSOA_2021.csv"
inputs["ASG_GB"] = "./input-data/Approximate_social_grade/Approximated_social_grade_LSOA_2021_GB.csv"

generated = {}
generated["EVCI-London"] = "./generated-files/EVCI-London.csv"
generated["input-analytics-db"] = "./generated-files/input-supply-demand-cleanmerge.csv"
generated["input-totals"] = "./generated-files/input-supply-demand-totals.csv"
generated["Mean_house_prices_London"] = "./generated-files/Mean_price_residential_properties_LSOA_1995_2022_London.csv"
generated["Median_house_prices_London"] = "./generated-files/Median_price_residential_properties_LSOA_1995_2022_London.csv"
generated["ASG_London"] = "./generated-files/Approximated_social_grade_LSOA_2021_London.csv"

outputs = {}
outputs["correlation_matrix_2021"] = "./output-data/correlation_matrix_2021.csv"

