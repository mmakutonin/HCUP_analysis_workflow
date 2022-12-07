# these functions are used whenever reading raw HCUP data text files
import pandas as pd
from utility_functions import pickle_file

data_dir = '../../raw_data/'

# Reads HCUP text data into dataframe
# Reference structure is {"col_name": [start_index, stop_index]}, see below
def read_data(reference, file_name):
    file_list = []
    with open(data_dir + file_name) as file:
        for count, row in enumerate(file):
            file_list.append({
                key: row[value[0]-1:value[1]] for key, value in reference.items()
            })
    return pd.DataFrame(file_list[2:]) #each HCUP file has 2 header rows

# Commented-out key-value pairs kept in case of future need
core_reference = {
    "sedd": {
        "dtypes": {
            "age":               "float",
            # "admission_type":    "float",
            # "weekend_admission": "float",
            # "cpt_codes":         ,
            "female":            "float",
            "homeless":          "float",
            # "ICD-10":            ,
            "record_id":         "float",
            "length_of_stay":    "float",
            "married":           "string",
            "race":              "float",
            "median_zip_income": "float",
            "payer":             "float",
            "visit_link":        "float",
            "total_charges":     "float",
            "year":              "float",
            "discharge_quarter": "float",
            "rural_urban":       "float"
        },
        "2018": {
            "age":               [1,3],
            # "admission_type":    [14,15],
            # "weekend_admission": [16,17],
            "cpt_codes":         [22,246],
            "female":            [640,641],
            "homeless":          [649,650],
            "ICD-10":            [660,1079],
            "record_id":         [1086,1100],
            "length_of_stay":    [1101,1105],
            "married":           [1113,1113],
            "race":              [1204,1205],
            "median_zip_income": [1262,1264],
            "rural_urban":       [1187,1188],
            "payer":             [1160,1161],
            "visit_link":        [1241,1249],
            "total_charges":     [1216,1225],
            "year":              [1250,1253],
            "discharge_quarter": [621,622]
        },
        "2017": {
            "age":               [1,3],
            # "admission_type":    [14,15],
            # "weekend_admission": [16,17],
            "cpt_codes":         [22,246],
            "female":            [643,644],
            "homeless":          [652,653],
            "ICD-10":            [663,872],
            "record_id":         [879,893],
            "length_of_stay":    [894,898],
            "married":           [906,906],
            "race":              [982,983],
            "median_zip_income": [1035,1037],
            "rural_urban":       [965,966],
            "payer":             [944,945],
            "visit_link":        [1019,1027],
            "total_charges":     [994,1003],
            "year":              [1028,1031],
            "discharge_quarter": [621,622]
        },
        "2016": {
            "age":               [1,3],
            # "admission_type":    [14,15],
            # "weekend_admission": [16,17],
            "cpt_codes":         [22,246],
            "female":            [643,644],
            "homeless":          [652,653],
            "ICD-10":            [663,865],
            "record_id":         [958,972],
            "length_of_stay":    [973, 977],
            "married":           [985, 985],
            "race":              [1061,1062],
            "median_zip_income": [1114, 1116],
            "rural_urban":       [1044,1045],
            "payer":             [1023,1024],
            "visit_link":        [1098,1106],
            "total_charges":     [1073,1082],
            "year":              [1107,1110],
            "discharge_quarter": [621,622]
        }
    },
    "sasd": {
        "dtypes": {
            "record_id":         "float",
            "visit_link":        "float",
            # "cpt_codes":         [22,246,45],
            # "ICD-10":            [663,1089],
            "payer":             "float",
            "year":              "float",
            "length_of_stay":    "float",
            "total_charges":   "float",
            "discharge_quarter": "float"
        },
        "2018": {
            "record_id":         [1096,1110],
            "visit_link":        [1251,1259],
            "cpt_codes":         [22,246],
            "ICD-10":            [663,1089],
            "payer":             [1170,1171],
            "year":              [1260,1263],
            "length_of_stay":    [1111,1115],
            "total_charges":   [1226,1235],
            "discharge_quarter": [621,622]
        },
        "2017": {
            "record_id":         [889,903],
            "visit_link":        [1029,1037],
            "cpt_codes":         [22,246],
            "ICD-10":            [666,882],
            "payer":             [954,955],
            "year":              [1038,1041],
            "length_of_stay":    [904,908],
            "total_charges":   [1004,1013],
            "discharge_quarter": [621,622]
        },
        "2016": {
            "record_id":         [968,982],
            "visit_link":        [1108,1116],
            "cpt_codes":         [22,246],
            "ICD-10":            [666,875],
            "payer":             [1033,1034],
            "year":              [1117,1120],
            "length_of_stay":    [983,987],
            "total_charges":   [1083,1092],
            "discharge_quarter": [621,622]
        }
    },
    "sid": {
        "dtypes": {
            "record_id":         "float",
            "visit_link":        "float",
            # "ICD-10-procedures": [726,1173],
            # "ICD-10":            [174,719],
            "payer":             "float",
            "year":              "float",
            "length_of_stay":    "float",
            "race":              "float",
            "total_charges":     "float",
            "discharge_quarter": "float",
            "ed_admission":      "float",
            "age":               "float",
            "married":           "string",
            "median_zip_income": "float",
            "rural_urban":       "float",
            "female":            "float",
            "homeless":          "float",
            "hospital_id":       "string"
        },
        "2018": {
            "record_id":         [1177,1191],
            "visit_link":        [1669,1677],
            "ICD-10-procedures": [726,1173],
            "ICD-10":            [174,719],
            "payer":             [1251,1252],
            "year":              [1678,1681],
            "length_of_stay":    [1192,1196],
            "race":              [1633 ,1634],
            "total_charges":     [1640,1649],
            "discharge_quarter": [54,55],
            "ed_admission":      [161,162],
            "age":               [1,3],
            "married":           [1204,1204],
            "median_zip_income": [1690,1692],
            "rural_urban":       [1278,1279],
            "female":            [159,160],
            "homeless":          [168,169],
            "hospital_id":       [64,80]
        },
        "2017": {
            "record_id":         [563,577],
            "visit_link":        [882,890],
            "ICD-10-procedures": [350,559],
            "ICD-10":            [127,343],
            "payer":             [637,638],
            "year":              [891,894],
            "length_of_stay":    [578,582],
            "race":              [846,847],
            "total_charges":     [853,862],
            "discharge_quarter": [52,53],
            "ed_admission":      [114,115],
            "age":               [1,3],
            "married":           [590,590],
            "median_zip_income": [898,900],
            "rural_urban":       [658,659],
            "female":            [112,113],
            "homeless":          [121,122],
            "hospital_id":       [62,78]
        },
        "2016": {
            "record_id":         [622,636],
            "visit_link":        [939,947],
            "ICD-10-procedures": [409,618],
            "ICD-10":            [135,344],
            "payer":             [696,697],
            "year":              [948,951],
            "length_of_stay":    [637,641],
            "race":              [905,906],
            "total_charges":     [910,919],
            "discharge_quarter": [52,53],
            "ed_admission":      [122,123],
            "age":               [1,3],
            "married":           [649,649],
            "median_zip_income": [955,957],
            "rural_urban":       [717,718],
            "female":            [120,121],
            "homeless":          [129,130],
            "hospital_id":       [62,78]
        }
    }
}

code_lengths = {
    "ICD-10": 7,
    "cpt_codes": 5,
    "ICD-10-procedures": 7
}

died_reference = {
    "dtypes": {
        "Died": "float",
        "visit_link": "float",
        "record_id": "float"
    },
    "sedd": {
        "2018": {
            "Died": [613,614],
            "visit_link": core_reference["sedd"]["2018"]["visit_link"],
            "record_id": core_reference["sedd"]["2018"]["record_id"]
        },
        "2017": {
            "Died": [613,614],
            "visit_link": core_reference["sedd"]["2017"]["visit_link"],
            "record_id": core_reference["sedd"]["2017"]["record_id"]
        },
        "2016": {
            "Died": [613,614],
            "visit_link": core_reference["sedd"]["2016"]["visit_link"],
            "record_id": core_reference["sedd"]["2016"]["record_id"]
        },
    },
    "sasd": {
        "2018": {
            "Died": [613,614],
            "visit_link": core_reference["sasd"]["2018"]["visit_link"],
            "record_id": core_reference["sasd"]["2018"]["record_id"]
        },
        "2017": {
            "Died": [613,614],
            "visit_link": core_reference["sasd"]["2017"]["visit_link"],
            "record_id": core_reference["sasd"]["2017"]["record_id"]
        },
        "2016": {
            "Died": [613,614],
            "visit_link": core_reference["sasd"]["2016"]["visit_link"],
            "record_id": core_reference["sasd"]["2016"]["record_id"]
        },
    },
    "sid": {
        "2018": {
            "Died": [46,47],
            "visit_link": core_reference["sid"]["2018"]["visit_link"],
            "record_id": core_reference["sid"]["2018"]["record_id"]
        },
        "2017": {
            "Died": [46,47],
            "visit_link": core_reference["sid"]["2017"]["visit_link"],
            "record_id": core_reference["sid"]["2017"]["record_id"]
        },
        "2016": {
            "Died": [46,47],
            "visit_link": core_reference["sid"]["2016"]["visit_link"],
            "record_id": core_reference["sid"]["2016"]["record_id"]
        },
    }
}

hospital_reference = {
    "sid": {
        "2018": {
            "DSHOSPID": [8,24],
            "HOSPID":   [25,29]
        },
        "2017": {
            "DSHOSPID": [8,24],
            "HOSPID":   [25,29]
        },
        "2016": {
            "DSHOSPID": [8,24],
            "HOSPID":   [25,29]
        }
    }
}