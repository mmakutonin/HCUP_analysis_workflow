# %%
import pandas as pd
import numpy as np
from utility_functions import load_file, pickle_file, starting_run, finished_run

# %%
codes = load_file("fully_filtered_codes.pickle")
summary_enhanced = load_file("summary_enhanced.pickle")
sid_costs = load_file("sid_costs.pickle")
sedd = load_file("filtered_sedd_data.pickle")
sid_ed = load_file("filtered_sid_ed_data.pickle")
sasd = load_file("filtered_sasd_data.pickle")
rvu_lookup = pd.read_csv(
    "../../raw_data/PPRRVU18_OCT.csv", skiprows=9, index_col=0
)

# %%
conv_factor=rvu_lookup.at["47563", "FACTOR"]
rvu_lookup = rvu_lookup[["DESCRIPTION", "TOTAL.1"]].dropna()

# %%
summary_enhanced["CPT Costs"] = codes.query("cpt_flag == True")\
.join(rvu_lookup, on="codes", how="left")["TOTAL.1"].reset_index()\
.groupby("visit_link").sum().mul(conv_factor)
summary_enhanced["CPT Costs"].fillna(0, inplace=True)
summary_enhanced["SID Costs"] = sid_costs.groupby('visit_link').sum().loc[:,'SID_costs']
summary_enhanced["SID Costs"].fillna(0, inplace=True)
summary_enhanced["Cost"] = summary_enhanced["SID Costs"].add(summary_enhanced["CPT Costs"])

# %%
sedd_sasd_costs = sedd[["discharge_quarter", "visit_link", "cpt_codes", "year"]]\
    .append(sasd[["discharge_quarter", "visit_link", "cpt_codes", "year"]])\
    .explode(column='cpt_codes').join(rvu_lookup, on='cpt_codes', how="left")\
    .groupby(['year', 'discharge_quarter', 'visit_link']).sum().mul(conv_factor)
costs_by_quarter = sedd_sasd_costs.join(
    sid_costs.groupby(['year', 'discharge_quarter', 'visit_link']).sum(),
    how="outer"
).fillna(0).sum(axis=1)
inpatient_admits_by_quarter = sid_costs.fillna(0).groupby(['year', 'discharge_quarter', 'visit_link']).count().iloc[:, 0]
ed_visits_by_quarter = pd.concat([
    sedd[["discharge_quarter", "visit_link", "year", "age"]],
    sid_ed[["discharge_quarter", "visit_link", "year", "age"]]
]).groupby(['year', 'discharge_quarter', 'visit_link']).count().iloc[:, 0]

# %%
outcomes_by_quarter = pd.DataFrame({
    "Cost": costs_by_quarter,
    "Inpatient Readmissions": inpatient_admits_by_quarter,
    "ED Readmissions": ed_visits_by_quarter
    }).fillna(0).reset_index().astype("float").groupby(['year', 'discharge_quarter', 'visit_link']).sum()\
    .reset_index().join(summary_enhanced[["Admitted", "initial_discharge_quarter", "initial_year"]], how='inner', on='visit_link')
outcomes_by_quarter["quarters_from_init"] = outcomes_by_quarter.aggregate(
    lambda row: (row['year']-row['initial_year'])*4+(row['discharge_quarter']-row['initial_discharge_quarter']),
    axis=1
)

# %%
#Correct for initial ED visit
outcomes_by_quarter["ED Readmissions"].update(
    outcomes_by_quarter.loc[
        outcomes_by_quarter['quarters_from_init'] == 0
    ]["ED Readmissions"]-1
)
#Correct for initial admission if admitted
outcomes_by_quarter["Inpatient Readmissions"].update(
    outcomes_by_quarter.loc[
        (outcomes_by_quarter['quarters_from_init'] == 0) &
        (outcomes_by_quarter['Admitted'])
    ]["Inpatient Readmissions"]-1
)

# %%
# Get rows that don't have an associated cost
missing_costs = sid_costs.set_index('visit_link')[sid_costs.set_index('visit_link')['SID_costs'].isna()].index
# Get rows that have a zero cost
zero_cost = summary_enhanced.query('Cost == 0').index
# If both conditions met, drop from summary table since it's a false zero
drop_rows = [index for index in missing_costs if index in zero_cost]
print(f"{len(drop_rows)} records dropped due to zero SID and SEDD costs.")
summary_enhanced.drop(drop_rows, inplace=True)

#filter outcomes_by_quarter and restrict columns
outcomes_by_quarter = outcomes_by_quarter.join(summary_enhanced, how="inner", on="visit_link", rsuffix="_x")\
    .groupby(["visit_link", "quarters_from_init"])[["Cost", "Inpatient Readmissions", "ED Readmissions"]].sum()

# %%
pickle_file("summary_costs_enhanced.pickle", summary_enhanced)
pickle_file("outcomes_by_quarter.pickle", outcomes_by_quarter)


