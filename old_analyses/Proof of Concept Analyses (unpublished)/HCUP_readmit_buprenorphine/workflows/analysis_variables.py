# This file contains variables that specify analysis constants needed to make the analysis pipeline applicable
# to multiple projects and research questions.

procedure_codes = { #currently cholecystectomy, change for new procedure
    "cpt_codes": [],
    "ICD-10-procedures": []
}
diagnosis_codes = [ #currently for opioid use disorder https://www.icd10data.com/ICD10CM/Index/D/Disorder#34418 https://s3.wp.wsu.edu/uploads/sites/2443/2019/09/DSM-5-Opioid-Use-Disorder-Diagnostic-Criteria.pdf 
    "F1110", "F1112", "F1113", "F1114", "F1115", "F1118", "F1119", #opioid abuse (mild)
    "F1120", "F1122", "F1123", "F1124", "F1125", "F1128", "F1129" #opioid dependence (severe)
    #excluding unspecified opioid use (F119)
]

def dataset_filtering_function(dataset_name, dataset_core, proc_code_type):
    return dataset_core[dataset_core["ICD-10"].transform(
        lambda x: any([code in x for code in diagnosis_codes])
    )].copy()


# Data Enrichment
# Separates records/patients into subgroups for statistical analysis. Currently surgery_type
de_col_keys = ["Suboxone Administered", "Suboxone Dose (mg)", "CC Overdose", "CC Withdrawal"]
de_col_values = {
    de_col_keys[0]: ["+ Suboxone", "- Suboxone"],
    de_col_keys[1]: [0,2,4,8,12], #based on available suboxone doses https://www.dynamed.com/drug-monograph/buprenorphine-naloxone#GUID-2B5708A8-E14B-4B17-8559-73C3995F8987
    de_col_keys[2]: ["Overdose", "Not Overdose"],
    de_col_keys[3]: ["Withdrawal", "No Withdrawal"],
    }
def data_enrichment_function(sedd, sasd, sid, sid_ed, codes, linker_table):
    code_types = {
        "Suboxone Administered": ["J0572", "J0573", "J0574", "J0575"],
        "Suboxone Dose (mg)": {"J0572": 2, "J0573": 4, "J0574": 8, "J0575": 12},
        "CC Overdose": ["F1112", "F1122"],
        "CC Withdrawal": ["F1113", "F1123"]
    }
    #Cholecystectomy Type Enrichment
    def join_linker_table(dataset):
        return linker_table.join(dataset, on="initial_record_id", how="inner", rsuffix="_x")
    def includes_code(dataset, code_col_name, new_col_name):
        return dataset.apply(
            lambda x: (any([code in x[code_col_name] for code in code_types[new_col_name]]), x["visit_link"]),
            axis=1,
            result_type='expand'
        ).groupby(1).max().rename(columns={0:new_col_name})

    linker_table[de_col_keys[0]] = includes_code(join_linker_table(sedd), 'cpt_codes', de_col_keys[0])
    linker_table[de_col_keys[0]] = linker_table[de_col_keys[0]].transform(
        lambda x: de_col_values[de_col_keys[0]][0] if x == True else de_col_values[de_col_keys[0]][1]
    )

    linker_table[de_col_keys[1]] = join_linker_table(sedd).apply(
        lambda x: (max([code if code in x['cpt_codes'] else "0" for code in code_types[de_col_keys[1]].keys()]), x["visit_link"]),
        axis=1,
        result_type='expand'
    ).groupby(1).max().loc[:,0].map(code_types[de_col_keys[1]])
    linker_table[de_col_keys[1]] = linker_table[de_col_keys[1]].fillna(0).astype("int")

    linker_table[de_col_keys[2]] = includes_code(join_linker_table(sedd), 'ICD-10', de_col_keys[2]).append(
        includes_code(join_linker_table(sid.loc[sid_ed.index]), 'ICD-10', de_col_keys[2])
    )
    linker_table[de_col_keys[2]] = linker_table[de_col_keys[2]].map({
        True: de_col_values[de_col_keys[2]][0],
        False: de_col_values[de_col_keys[2]][1]
    })

    linker_table[de_col_keys[3]] = includes_code(join_linker_table(sedd), 'ICD-10', de_col_keys[3]).append(
        includes_code(join_linker_table(sid.loc[sid_ed.index]), 'ICD-10', de_col_keys[3])
    )
    linker_table[de_col_keys[3]] = linker_table[de_col_keys[3]].map({
        True: de_col_values[de_col_keys[3]][0],
        False: de_col_values[de_col_keys[3]][1]
    })

    return linker_table

logreg_targets = {
    'Given Discharge - Buprenorphine vs None': \
        lambda dataset: dataset.loc[
            dataset['Admitted'] == True,
            de_col_keys[0]
        ].eq(de_col_values[de_col_keys[0]][0]),
    'Admission vs Discharge': \
        lambda dataset: dataset['Admitted']
}

outcome_cols = [
    "Cost",
    "Inpatient Readmissions",
    "ED Readmissions",
    "Died",
]