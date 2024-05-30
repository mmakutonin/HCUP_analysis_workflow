# This file contains variables that specify analysis constants needed to make the analysis pipeline applicable
# to multiple projects and research questions.

procedure_codes = { # no procedure
    "cpt_codes": [],
    "ICD-10-procedures": []
}
diagnosis_codes = [ #currently cannabis hyperemesis syndrome : vomiting + cannabis use 
    "R1110", ["F1210", "F122", "F1220", "F129", "F1290"]
]

def dataset_filtering_function(dataset, proc_code_type):
    return dataset[dataset["ICD-10"].transform(
        lambda x: all([diagnosis_codes[0] in x, any([code in x for code in diagnosis_codes[1]])])
    )].copy()


# Data Enrichment
# Separates records/patients into subgroups for statistical analysis. Currently surgery_type
de_col_name = "hyperemesis"
de_col_values = [
    "Emergency Surgery",
    "ED Readmit Surgery",
    "Delayed Surgery",
    "No Surgery"
]
def data_enrichment_function(sedd, sasd, sid, sid_ed, linker_table):
    def join_linker_table(dataset):
        return dataset.join(linker_table, on="visit_link", how="inner", rsuffix="_x")
    def label_surg_type(dataset, procedure_code_name, new_col_name):
        return dataset.apply(
            lambda x: (any([code in x[procedure_code_name] for code in procedure_codes[procedure_code_name]]), x["visit_link"]),
            axis=1,
            result_type='expand'
        ).groupby(1).max().rename(columns={0:new_col_name})
    
    linker_table[de_col_name] = sedd.append(sid.loc[sid_ed.index]).loc[linker_table["initial_record_id"]].apply(
        lambda x: (any([code in x["cpt_codes"] for code in procedure_codes["cpt_codes"]])\
                   if isinstance(x["ICD-10-procedures"], float) else \
                   any([code in x["ICD-10-procedures"] for code in procedure_codes["ICD-10-procedures"]]),
                   x["visit_link"]), #float = NaN, otherwise would be a list of codes
            axis=1,
            result_type='expand'
        ).groupby(1).max().rename(columns={0:"emergency_surgeries"}).join([
        label_surg_type(join_linker_table(sid),"ICD-10-procedures","sid_surgeries"),
        label_surg_type(join_linker_table(sasd),"cpt_codes","sasd_surgeries"),
        label_surg_type(join_linker_table(sedd),"cpt_codes","delayed_emergency_surgeries"),
        label_surg_type(join_linker_table(sid.loc[sid_ed.index]),"ICD-10-procedures","delayed_emergency_sid_surgeries"),
     ]).apply(
        lambda row: de_col_values[0] if row["emergency_surgeries"] == True\
        else de_col_values[1] if row["delayed_emergency_surgeries"] == True or row["delayed_emergency_sid_surgeries"] == True\
        else de_col_values[2] if row["sasd_surgeries"] == True or row["sid_surgeries"] == True else de_col_values[3],
        axis=1
    )
    
    return linker_table

logreg_targets = {
    'Surgery vs No Surgery': lambda dataset: dataset.loc[dataset.index, de_col_name].eq(de_col_values[3]),
    'Given Surgery - Emergency vs Delayed': lambda dataset: dataset.loc[dataset[de_col_name].ne(de_col_values[3]), de_col_name].eq(de_col_values[0]),
    'Emergency Surgery vs Others': lambda dataset: dataset.loc[dataset.index, de_col_name].eq(de_col_values[0]),
    'Given ED Discharge - Surgery vs No Surgery': lambda dataset: dataset.loc[dataset[de_col_name].ne(de_col_values[0]), de_col_name].ne(de_col_values[3])
}