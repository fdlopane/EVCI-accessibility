"""
Analysis of the data collected from the EVCI project.

Author Fulvio D. Lopane
Centre for Advanced Spatial Analysis

started coding: November 2024
"""
from unittest.mock import inplace

import pandas as pd
from config import *
import os

if not os.path.exists(generated["input-analytics-db"]):
    # Import EVCI counts at lsoa level
    EVCI = pd.read_csv(inputs["EVCI-LSOA"])

    # Import EV licensing counts at lsoa level
    # Note 338 out of 4994 (6.8%) LSOAs have null values, becasue small areas which have never had 5 or more vehicles
    # in scope are excluded from these datasets. --> Fill with 0?
    EV_counts = pd.read_csv(inputs["EV-counts-LSOA"])
    EV_counts.fillna(0, inplace=True)

    # Merge the two datafames
    analysis_df =  EVCI.merge(EV_counts, on='LSOA21CD')
    # Rename and drop duplicate columns
    # NOTE delete ALL dashes (-), spaces, and underscores (_) from column names
    analysis_df.drop(columns=['LSOA21NM_y', 'GlobalID_y'], inplace=True)
    analysis_df.rename(columns={'LSOA21NM_x': 'LSOA21NM',
                                'GlobalID_x': 'GlobalID',
                                '_Fuel': 'Fuel',
                                '_Keepershi': 'Keepership',
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
                                '_2011 Q4': 'y2011Q4'}, inplace=True)

    # Remove EV licensing outliers (remove those LSOAs with licensing > lic_threshold)
    lic_threshold = 500
    analysis_df = analysis_df.drop(analysis_df[analysis_df.y2024Q2 > lic_threshold].index)

    # Save the cleaned data
    analysis_df.to_csv(generated["input-analytics-db"], index=False)
else:
    analysis_df = pd.read_csv(generated["input-analytics-db"])

if not os.path.exists(generated["input-totals"]):
    # List the columns I want to calculate the sum of
    fields = ["EVCI2024", "EVCI2023", "EVCI2022", "EVCI2021", "EVCI2020", "EVCI2019", "EVCI2018", "EVCI2017", "EVCI2015", "EVCI2013", "EVCI2012", "y2024Q1", "y2024Q2", "y2023Q4", "y2023Q3", "y2023Q2", "y2023Q1", "y2022Q4", "y2022Q3", "y2022Q2", "y2022Q1", "y2021Q4", "y2021Q3", "y2021Q2", "y2021Q1", "y2020Q4", "y2020Q3", "y2020Q2", "y2020Q1", "y2019Q4", "y2019Q3", "y2019Q2", "y2019Q1", "y2018Q4", "y2018Q3", "y2018Q2", "y2018Q1", "y2017Q4", "y2017Q3", "y2017Q2", "y2017Q1", "y2016Q4", "y2016Q3", "y2016Q2", "y2016Q1", "y2015Q4", "y2015Q3", "y2015Q2", "y2015Q1", "y2014Q4", "y2014Q3", "y2014Q2", "y2014Q1", "y2013Q4", "y2013Q3", "y2013Q2", "y2013Q1", "y2012Q4", "y2012Q3", "y2012Q2", "y2012Q1", "y2011Q4"]
    fields.sort()

    statistics = pd.DataFrame()

    for f in fields:
        tot = analysis_df[f].sum()
        statistics[f + "_tot"] = [tot]

    # Save the statistics
    statistics.to_csv(generated["input-totals"], index=False)