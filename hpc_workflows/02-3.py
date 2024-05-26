import pandas as pd
import numpy as np
from data_reading_functions import read_data, hospital_reference, data_dir
from utility_functions import load_file, pickle_file

sid_inflation_adjustment = pd.DataFrame([
    #needed for 2.3, based on annual inflation data from Bureau of Labor Statistics
    #https://data.bls.gov/pdq/SurveyOutputServlet
    {'year':2016, 'discharge_quarter': 1, 'inflation_adjustment': 237.111/251.107},
    {'year':2016, 'discharge_quarter': 2, 'inflation_adjustment': 240.229/251.107},
    {'year':2016, 'discharge_quarter': 3, 'inflation_adjustment': 240.849/251.107},
    {'year':2016, 'discharge_quarter': 4, 'inflation_adjustment': 241.353/251.107},
    {'year':2017, 'discharge_quarter': 1, 'inflation_adjustment': 243.603/251.107},
    {'year':2017, 'discharge_quarter': 2, 'inflation_adjustment': 244.733/251.107},
    {'year':2017, 'discharge_quarter': 3, 'inflation_adjustment': 245.519/251.107},
    {'year':2017, 'discharge_quarter': 4, 'inflation_adjustment': 246.669/251.107},
    {'year':2018, 'discharge_quarter': 1, 'inflation_adjustment': 248.991/251.107},
    {'year':2018, 'discharge_quarter': 2, 'inflation_adjustment': 251.588/251.107},
    {'year':2018, 'discharge_quarter': 3, 'inflation_adjustment': 252.146/251.107},
    {'year':2018, 'discharge_quarter': 4, 'inflation_adjustment': 252.038/251.107},
    #remainder from https://www.bls.gov/cpi/tables/supplemental-files/historical-cpi-u-202403.pdf
    {'year':2019, 'discharge_quarter': 1, 'inflation_adjustment': (251.712+252.776+254.202)/3/251.107},
    {'year':2019, 'discharge_quarter': 2, 'inflation_adjustment': (255.548+256.092+256.143)/3/251.107},
    {'year':2019, 'discharge_quarter': 3, 'inflation_adjustment': (256.571+256.558+256.759)/3/251.107},
    {'year':2019, 'discharge_quarter': 4, 'inflation_adjustment': (257.346+257.208+256.974)/3/251.107},
    {'year':2020, 'discharge_quarter': 1, 'inflation_adjustment': (257.971+258.678+258.115)/3/251.107},
    {'year':2020, 'discharge_quarter': 2, 'inflation_adjustment': (256.381+256.389+257.797)/3/251.107},
    {'year':2020, 'discharge_quarter': 3, 'inflation_adjustment': (259.101+259.918+260.280)/3/251.107},
    {'year':2020, 'discharge_quarter': 4, 'inflation_adjustment': (260.388+260.229+260.474)/3/251.107},
    {'year':2021, 'discharge_quarter': 1, 'inflation_adjustment': (261.582+263.014+264.877)/3/251.107},
    {'year':2021, 'discharge_quarter': 2, 'inflation_adjustment': (267.054+269.195+271.696)/3/251.107},
    {'year':2021, 'discharge_quarter': 3, 'inflation_adjustment': (273.003+273.567+274.310)/3/251.107},
    {'year':2021, 'discharge_quarter': 4, 'inflation_adjustment': (276.589+277.948+278.802)/3/251.107},
]).set_index(['year', 'discharge_quarter'])

def enrich_sid_costs(analysis_name:str):
    hospital_files = []
    for year in [2016, 2017, 2018, 2019, 2020, 2021]:
        data_file = read_data(hospital_reference["sid"][str(year)], f"MD_SID_{year}_AHAL.asc")
        data_file["year"] = year
        hospital_files.append(data_file)
    hospital_lookup = pd.concat(hospital_files).astype('int').set_index(['HOSPID', 'year'])
    ccr_list = []
    # uses APICC for hospitals for which it is available, and GAPICC for those which APICC is not available.
    for ccr_name in ['cc2016CD', 'cc2017CD_v2', 'cc2018CDSID_v2', 'cc2019CDSID', 'cc2020CDSID']:
        ccr = pd.read_csv(f'{data_dir}{ccr_name}.csv', index_col="'HOSPID'")
        ratios = ccr["'APICC'"].str.strip().replace(".", np.nan)\
        .combine_first(ccr["'GAPICC'"]).astype('float').to_frame()
        ratios['year'] = ccr.iat[0,0]
        ccr_list.append(ratios.copy())
        if '2020' in ccr_name: #workaround since we don't have a 2021 CCR file; TODO FIX THIS
            ratios['year'] = 2021
            ccr_list.append(ratios.copy())
    ccr = pd.concat(ccr_list).reset_index()
    ccr['HOSPID'] = ccr["'HOSPID'"].transform(lambda hospid: hospid[1:-1]).astype('int')
    ccr['year'] = ccr['year'].astype('int')
    ccr = ccr.join(hospital_lookup, on=["HOSPID", "year"], how="inner").loc[:, ["'APICC'", 'year', 'DSHOSPID']]

    sid_core_filtered = load_file("filtered_sid_data.pickle", analysis_name)\
        [['hospital_id', 'year', 'total_charges', 'visit_link', 'discharge_quarter']].astype('int')\
        .join(ccr.set_index(["DSHOSPID", "year"]), on=['hospital_id', 'year'], how="left")\
        .join(sid_inflation_adjustment, on=['year', 'discharge_quarter'], how='left')
    sid_core_filtered['SID_costs'] = sid_core_filtered['total_charges'].mul(sid_core_filtered["'APICC'"])\
        .mul(sid_core_filtered['inflation_adjustment'])
    pickle_file("sid_costs.pickle", analysis_name, sid_core_filtered.loc[:, ['visit_link', 'SID_costs', 'discharge_quarter', 'year']])
