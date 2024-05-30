# %%
import pandas as pd
from utility_functions import load_file, pickle_file, starting_run, finished_run

# %%
sid_core_filtered = load_file("sid_core_filtered.pickle")

# %%
sedd_appendix = sid_core_filtered.loc[sid_core_filtered["ed_admission"] > 0]

# %%
sedd_appendix_constrained = sedd_appendix.loc[:,[
    'visit_link',
    'payer',
    'year',
    'age',
    'married',
    'race',
    'median_zip_income',
    'discharge_quarter',
    'female',
    'homeless'
]]
sedd_appendix_constrained['total_charges'] = 0
sedd_appendix_constrained['ICD-10'] = sedd_appendix['ICD-10'].transform(lambda x: [f"{x[0][0:3]}-tmp"])

# %%
pickle_file("sedd_appendix.pickle", sedd_appendix_constrained)

# %%



