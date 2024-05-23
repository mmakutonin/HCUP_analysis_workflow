# This file contains variables that specify analysis constants needed to make the analysis pipeline applicable
# to multiple projects and research questions.
import pandas as pd
import numpy as np

procedure_codes = { #currently cholecystectomy, change for new procedure
    "cpt_codes": ["47562","47563","47564","47600","47605","47610","47612","47620"],
    "ICD-10-procedures": ["0FT40ZZ", "0FT44ZZ", "0FB40ZX", "0FB40ZZ", "0FB43ZX", "0FB43ZZ", "0FB44ZX", "0FB44ZZ", "0FB48ZX", "0FB48ZZ"]
}
diagnosis_codes = [ #currently for cholelithiosis/biliary colic
    "K802", "K805", "K807", "K808"
]

code_category_dict = {
    #Charlson Comorbidity Index variables (needed for 2.0), ICD-10 code definitions from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6684052
    "Myocardial Infarction History": ["I21", "I22", "I252"],
    "Heart Failure": ["I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427", "I428", "I429", "I43", "I50", "P290"],
    "Peripheral Vascular Disease": ["I70", "I71", "I731", "I738", "I739", "I771", "I790", "I791", "I798", "K551", "K558", "K559", "Z958", "Z959"],
    "CVA/TIA": ["G45", "G46", "H340", "H341", "H342", "I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68"],
    "Dementia": ["F01", "F02", "F03", "F04", "F05", "F061", "F068", "G132", "G138", "G30", "G310", "G311", "G312", "G914", "G94", "R4181", "R54"],
    "COPD": ["J4", "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67", "J684", "J701", "J703"],
    "Rheumatic Disease": ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"],
    "Peptic Ulcer Disease": ["K25", "K26", "K27", "K28"],
    "Mild Liver Disease": ["B18", "K700", "K701", "K702", "K703", "K709", "K713", "K714", "K715", "K717", "K73", "K74", "K760", "K762", "K763", "K764", "K768", "K769", "Z944"],
    "Severe Liver Disease": ["I850", "I864", "K704", "K711", "K721", "K729", "K765", "K766", "K767"],
    "Uncomplicated Diabetes": ["E080", "E081", "E086", "E088", "E089", "E090", "E091", "E096", "E098", "E099", "E100", "E101", "E106", "E108", "E109", "E110", "E111", "E116", "E118", "E119", "E130", "E131", "E136", "E138", "E139"],
    "Complicated Diabetes": ["E082", "E083", "E084", "E085", "E092", "E093", "E094", "E095", "E102", "E103", "E104", "E105", "E112", "E113", "E114", "E115", "E132", "E133", "E134", "E135"],
    "Hemiplegia": ["G041", "G114", "G800", "G801", "G802", "G81", "G82", "G83"],
    "Uncomplicated Renal Disease": ["I129", "I130", "I1310", "N03", "N05", "N181", "N182", "N183", "N184", "N189", "Z940"],
    "Severe Renal Disease": ["I120", "I1311", "N185", "N186", "N19", "N250", "Z49", "Z992"],
    "Malignancy": ["C"], #some will be superceded by invalid malignancies to save processing time
    "Metastatic Tumor": ["C77", "C78", "C79", "C800", "C802"],
    "Invalid Malignancy": ["C44", "C4A", "C64", "C65", "C66", "C67", "C68", "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C7A", "C7B", "C86"],
    "HIV": ["B20"],
    "AIDS Opportunistic Infection": ["A021", "A072", "A073", "A1", "A31", "A812", "B00", "B25", "B37", "B38", "B39", "B45", "B58", "B59", "C53", "C46", "C8", "C9", "G934", "R64", "Z8701"],
    
    #Additional comorbidities
    "systemic hypertension":["I10", "I11", "I12", "I13", "I14", "I15", "I16", "O10", "O11", "O13", "O14", "O15", "O16"],
    "obesity": ["O9921", "O66", "Z683", "Z684"],
    "aspirin": ["Z7982"],
    "nicotine dependence": ["F17", "T652", "Z87891", "Z720"],
    "hyperlipidemia": ["E780", "E781", "E782", "E783", "E784", "E785"],
    "diabetes mellitus": ["E08", "E09", "E10", "E11", "E13", "O24", "Z8632"],
    "ischemic heart disease": ["I20", "I21", "I22", "I23", "I24", "I25"],
    "alcohol-related disorders": ["F10"],
    "mood disorders": ["F2", "F3", "F4"],

    #Analysis-specific comorbidities
    "pregnant": ["O", "Z33", "Z34", "Z3A"]
}

def dataset_filtering_function(dataset_name, dataset_core, proc_code_type):
    if dataset_name == "sasd":
        return dataset_core[dataset_core[proc_code_type].apply(
            lambda x: any([code in x for code in procedure_codes[proc_code_type]])
        )].copy()
    else:
        return dataset_core[dataset_core["ICD-10"].transform(
            lambda x: any([x.startswith(code) for code in diagnosis_codes])
        )].copy()


# Data Enrichment
# Separates records/patients into subgroups for statistical analysis. Currently surgery_type
de_col_keys = [
    "Admission Status",
    "Biliary Colic Type",
    "Cholecystectomy Type",
    "Obtained Cholecystectomy",
    "Post-Index Visit Cholecystitis",
    "Initial Surgical Management"
]
de_col_values = {
    de_col_keys[0]: ["Admitted", "Not Admitted"],
    de_col_keys[1]: ["Complicated", "Uncomplicated"],
    de_col_keys[2]: [
        "Immediate",
        "Return Emergency",
        "Delayed",
        "No Surgery"
        ],
    de_col_keys[3]: ["Obtained Surgery", "Did Not Obtain Surgery"],
    de_col_keys[4]: ["New Cholecystitis", "No New Cholecystitis"],
    de_col_keys[5]: ["Immediate Cholecystectomy Obtained", "Initial Management Without Cholecystectomy"]
    }
def data_enrichment_function(sedd, sasd, sid, sid_ed, codes, linker_table):
    code_types = {
        de_col_keys[1]: ["K800", "K801", "K803", "K804", "K806", "K81", "K851", "K82A"],
        de_col_keys[2]: [*procedure_codes["cpt_codes"], *procedure_codes["ICD-10-procedures"]]
    }
    def code_linker(code_type_key, init_visit="any", ed_visit="any",codes=codes):
        if init_visit != "any":
            codes = codes.query(f"init_chart == {init_visit}")
        if ed_visit != "any":
            codes = codes.query(f"ed_flag == {ed_visit}")
        return codes.reset_index().\
            groupby("visit_link")["codes"].unique().apply(lambda x: [st.strip() for st in x]).\
            transform(lambda x: any([code in str(x) for code in code_types[code_type_key]]))
    
    #Label Surgery Type
    linker_table[de_col_keys[2]] = code_linker(de_col_keys[2]).map({ #see if any surgery
        True: de_col_values[de_col_keys[2]][2],
        False: de_col_values[de_col_keys[2]][3]
    })
    linker_table[de_col_keys[2]].update(code_linker(de_col_keys[2], True, True).map({ #see if surgery is immediate
        True: de_col_values[de_col_keys[2]][0],
        False: np.nan
    }))
    linker_table[de_col_keys[2]].update(code_linker(de_col_keys[2], False, True).map({ #see if surgery is return emergency
        True: de_col_values[de_col_keys[2]][1],
        False: np.nan
    }))

    #Obtained Surgery
    linker_table[de_col_keys[3]] = code_linker(de_col_keys[2]).map({ #see if any surgery
        True: de_col_values[de_col_keys[3]][0],
        False: de_col_values[de_col_keys[3]][1]
    })

    # Admitted: modify admits to include patients that got emergency surgery
    linker_table[de_col_keys[0]] = (linker_table["Admitted"] | linker_table[de_col_keys[2]]\
        .eq(de_col_values[de_col_keys[2]][0])).map({
            True: de_col_values[de_col_keys[0]][0],
            False: de_col_values[de_col_keys[0]][1]
        })

    #Biliary Colic Type
    linker_table[de_col_keys[1]] = code_linker(de_col_keys[1],True,True).map({
        True: de_col_values[de_col_keys[1]][0],
        False: de_col_values[de_col_keys[1]][1]
    })

    # New Cholecystitis
    linker_table[de_col_keys[4]] = code_linker(de_col_keys[1],False).map({
        True: de_col_values[de_col_keys[4]][0],
        False: de_col_values[de_col_keys[4]][1]
    })
    linker_table[de_col_keys[4]] = linker_table[de_col_keys[4]].fillna(de_col_values[de_col_keys[4]][1]) # since not all visit_links captured (since looking only at revisits)

    # Immediate Cholecystectomy
    linker_table[de_col_keys[5]] = linker_table[de_col_keys[2]].eq(de_col_values[de_col_keys[2]][0]).map({
        True: de_col_values[de_col_keys[5]][0],
        False: de_col_values[de_col_keys[5]][1]
    })
    
    print(linker_table["Biliary Colic Type"].value_counts())
    
    return linker_table

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
]).set_index(['year', 'discharge_quarter'])

demographic_tables = [
    {
        "key": de_col_keys[0],
        "query_string": "Cost >= 0", #cost should always be positive, making this a universal filter
        "save_filepath": f"../tables/Table 1 {de_col_keys[0]}.csv"
    },
    {
        "key": de_col_keys[0],
        "query_string": "Complicated == False",
        "save_filepath": f"../tables/Table 1 Uncomplicated {de_col_keys[0]}.csv"
    },
    {
        "key": de_col_keys[5],
        "query_string": "Complicated == False and Admitted == True",
        "save_filepath": f"../tables/Table 1 Uncomplicated Admitted {de_col_keys[5]}.csv"
    },
    {
        "key": de_col_keys[3],
        "query_string": "Complicated == False and Admitted == False",
        "save_filepath": f"../tables/Table 1 Uncomplicated Unadmitted {de_col_keys[3]}.csv"
    },
    {
        "key": de_col_keys[5],
        "query_string": "Complicated == True",
        "save_filepath": f"../tables/Table 1 Complicated {de_col_keys[5]}.csv"
    },
    {
        "key": de_col_keys[2],
        "query_string": "Complicated == True and not `Immediate`",
        "save_filepath": f"../tables/Table 1 Complicated Non-immediate {de_col_keys[2]}.csv"
    }

]

logreg_targets = {
    'Immediate Cholecystectomy vs Others for Complicated Colic': \
        lambda dataset: dataset.loc[
            dataset[de_col_keys[1]] == de_col_values[de_col_keys[1]][0],
            de_col_keys[2]
        ].eq(de_col_values[de_col_keys[2]][0]),
    'No Surgery vs Others for Complicated Colic': \
        lambda dataset: dataset.loc[
            (dataset[de_col_keys[1]] == de_col_values[de_col_keys[1]][0]) &
            (dataset[de_col_keys[2]] != de_col_values[de_col_keys[2]][0]),
            de_col_keys[2]
        ].eq(de_col_values[de_col_keys[2]][3]),
    'Discharge vs Admission for Uncomplicated Colic':\
        lambda dataset: dataset.loc[
            dataset[de_col_keys[1]] == de_col_values[de_col_keys[1]][1],
            de_col_keys[0]
        ].eq(de_col_values[de_col_keys[0]][1]),
    'Given Discharge and Uncomplicated Colic - No Surgery vs Others':\
        lambda dataset: dataset.loc[
            (dataset[de_col_keys[1]] == de_col_values[de_col_keys[1]][1]) & \
            (dataset[de_col_keys[0]] == de_col_values[de_col_keys[0]][1]),
            de_col_keys[2]
        ].eq(de_col_values[de_col_keys[2]][3]),
    'Given Admission and Uncomplicated Colic - Immediate Surgery vs Others':\
        lambda dataset: dataset.loc[
            (dataset[de_col_keys[1]] == de_col_values[de_col_keys[1]][1]) & \
            (dataset[de_col_keys[0]] == de_col_values[de_col_keys[0]][0]),
            de_col_keys[2]
        ].eq(de_col_values[de_col_keys[2]][0])
}

outcome_cols = [
    "Cost",
    "Inpatient Readmissions",
    "ED Readmissions",
    "Died",
    "LOS"
    # "Obtained Cholecystectomy",
    # "Post-Index Visit Cholecystitis"
]

chart_plotting = [
    #complicated
    {
        'metric':'Cost',
        'y_axis_label': 'Cost (USD)',
        'title': 'Cost of Immediate Cholecystectomy vs Non-Surgical Management Over Time',
        'true_label': 'Immediate Cholecystectomy',
        'false_label': 'Non-Surgical Management',
        'logreg_key': 'Immediate Cholecystectomy vs Others for Complicated Colic',
    },
    {
        'metric':'ED Readmissions',
        'y_axis_label': 'Repeat ED Visits per Patient',
        'title': 'Repeat ED Visit Rates for Immediate Cholecystectomy vs Non-Surgical Management Over Time',
        'true_label': 'Immediate Cholecystectomy',
        'false_label': 'Non-Surgical Management',
        'logreg_key': 'Immediate Cholecystectomy vs Others for Complicated Colic',
    },
    {
        'metric':'Inpatient Readmissions',
        'y_axis_label': 'Repeat Hospitalizations per Patient',
        'title': 'Repeat Hospitalization Rates for Immediate Cholecystectomy vs Non-Surgical Management Over Time',
        'true_label': 'Immediate Cholecystectomy',
        'false_label': 'Non-Surgical Management',
        'logreg_key': 'Immediate Cholecystectomy vs Others for Complicated Colic',
    },
    #uncomplicated
    {
        'metric':'Cost',
        'y_axis_label': 'Cost (USD)',
        'title': 'Cost of Initial Admission vs Discharge Management Over Time',
        'true_label': 'Discharged on Index Visit',
        'false_label': 'Admitted on Index Visit',
        'logreg_key': 'Discharge vs Admission for Uncomplicated Colic',
        'forecast_length': 30
    },
    {
        'metric':'ED Readmissions',
        'y_axis_label': 'Repeat ED Visits per Patient',
        'title': 'Repeat ED Visit Rates of Initial Admission vs Discharge Management Over Time',
        'true_label': 'Discharged on Index Visit',
        'false_label': 'Admitted on Index Visit',
        'logreg_key': 'Discharge vs Admission for Uncomplicated Colic',
    },
    {
        'metric':'Inpatient Readmissions',
        'y_axis_label': 'Repeat Hospitalizations per Patient',
        'title': 'Repeat Hospitalization Rates of Initial Admission vs Discharge Management Over Time',
        'true_label': 'Discharged on Index Visit',
        'false_label': 'Admitted on Index Visit',
        'logreg_key': 'Discharge vs Admission for Uncomplicated Colic',
    },
]