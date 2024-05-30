from importlib import import_module
import sys

'''
Running this file requires a runtime argument pointing to the right analysis_variables to use.
i.e. python3 main.py appendicitis, where appendicitis would point to appendicitis_analysis_variables.py
and create corresponding data folders in ../results/appendicitis/
'''
analysis_name = sys.argv[1]
analysis_variables = import_module(f"{analysis_name}_analysis_variables")

# Run all workflow functions
# import_module('01-0').import_data(analysis_name, analysis_variables.dataset_filtering_function)
# import_module('01-1').create_sedd_appendix(analysis_name)
import_module('02-0').create_linked_datasets(
    analysis_name,
    analysis_variables.linker_table_filtering_function,
    analysis_variables.data_enrichment_function,
    analysis_variables.code_category_dict,
    analysis_variables.init_visit_datasets
)
import_module('02-1').note_deaths_in_study_period(analysis_name)
import_module('02-2').convert_data_values_to_readable_format(analysis_name)
import_module('02-3').enrich_sid_costs(analysis_name)
import_module('02-4').enrich_care_costs_by_quarter(analysis_name)
import_module('03-1').create_summary_tables(
    analysis_name,
    analysis_variables.demographic_table_configurations,
    analysis_variables.code_category_dict,
    analysis_variables.de_col_keys,
    analysis_variables.de_col_values
)
import_module('03-2').run_multivariate_analyses(
    analysis_variables.outcome_cols,
    analysis_variables.code_category_dict,
    analysis_variables.logreg_targets,
    analysis_variables.linreg_targets,
    analysis_name,
    analysis_variables.features_to_remove if 'features_to_remove' in dir(analysis_variables) else [],
    include_pca=True
)
import_module('03-3').plot_figures(
    analysis_name,
    analysis_variables.chart_plotting_configurations,
    analysis_variables.logreg_targets
)
