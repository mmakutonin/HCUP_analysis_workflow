from importlib import import_module
import sys

'''
Running this file requires a runtime argument pointing to the right analysis_variables to use.
i.e. python3 main-regenerate-outputs-only.py appendicitis, where appendicitis would point to appendicitis_analysis_variables.py
and create corresponding data folders in ../results/appendicitis/
'''
analysis_name = sys.argv[1]
analysis_variables = import_module(f"{analysis_name}_analysis_variables")

# Run only workflow functions associated with results presentation
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
    include_pca=True
)
import_module('03-3').plot_figures(
    analysis_name,
    analysis_variables.chart_plotting_configurations,
    analysis_variables.logreg_targets
)
