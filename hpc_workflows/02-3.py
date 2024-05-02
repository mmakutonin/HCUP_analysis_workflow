# %%
import pandas as pd
import numpy as np
from data_reading_functions import data_dir, read_data, hospital_reference
from utility_functions import load_file, pickle_file
from analysis_variables import sid_inflation_adjustment

# %%
hospital_files = []
for year in [2016, 2017, 2018, 2019, 2020, 2021]:
    data_file = read_data(hospital_reference["sid"][str(year)], f"MD_SID_{year}_AHAL.asc")
    data_file["year"] = year
    hospital_files.append(data_file)
hospital_lookup = pd.concat(hospital_files).astype('int').set_index(['HOSPID', 'year'])

# %%
ccr_list = []
# uses APICC for hospitals for which it is available, and GAPICC for those which APICC is not available.
for ccr_name in ['cc2016CD', 'cc2017CD_v2', 'cc2018CDSID_v2', 'cc2019CDSID', 'cc2020CDSID']:
    ccr = pd.read_csv(f'../../raw_data/{ccr_name}.csv', index_col="'HOSPID'")
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

# %%
sid_core_filtered = load_file("filtered_sid_data.pickle")\
[['hospital_id', 'year', 'total_charges', 'visit_link', 'discharge_quarter']].astype('int')\
.join(ccr.set_index(["DSHOSPID", "year"]), on=['hospital_id', 'year'], how="left")\
.join(sid_inflation_adjustment, on=['year', 'discharge_quarter'], how='left')

# %%
sid_core_filtered['SID_costs'] = sid_core_filtered['total_charges'].mul(sid_core_filtered["'APICC'"])\
    .mul(sid_core_filtered['inflation_adjustment'])

# %%
pickle_file("sid_costs.pickle", sid_core_filtered.loc[:, ['visit_link', 'SID_costs', 'discharge_quarter', 'year']])
