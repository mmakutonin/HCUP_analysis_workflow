import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from utility_functions import load_file
from scipy.stats import norm

# %%
plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=22)     # fontsize of the axes title
plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('axes', titlesize=28)  # fontsize of the figure title

def transform_data_to_quarterly_outcomes(analysis_name:str):
    if not os.path.isdir(f"../results/{analysis_name}/figures/comparison plots"):
        os.mkdir(f"../results/{analysis_name}/figures/comparison plots")
    outcomes = load_file("outcomes_by_quarter.pickle", analysis_name)
    filtered_data = load_file("summary_costs_enhanced.pickle", analysis_name)
    category_status = load_file("comorbidities.pickle", analysis_name)
    summary_table = filtered_data.join(category_status, how="left")
    empty_outcomes = pd.DataFrame({
            'visit_link': np.repeat(summary_table.index, 5),
            'quarters_from_init': np.tile([0,1,2,3,4], summary_table.index.size),
            "Cost": np.zeros(5*summary_table.index.size),
            "Inpatient Readmissions": np.zeros(5*summary_table.index.size),
            "ED Readmissions": np.zeros(5*summary_table.index.size)
        }).set_index(['visit_link','quarters_from_init'])
    empty_outcomes.update(
        outcomes
    )
    return filtered_data, empty_outcomes.reset_index().groupby('visit_link')\
        [['quarters_from_init', "Cost", "Inpatient Readmissions", "ED Readmissions"]]\
        .rolling(5, on='quarters_from_init', min_periods=1).sum().dropna()\
        .reset_index().set_index('visit_link')

def plot_chart(
    analysis_name,
    filtered_data,
    rolling_outcomes,
    logreg_targets,
    metric,
    y_axis_label,
    title,
    true_label,
    false_label,
    logreg_key,
    forecast_length = 10,
    fig_size = (18, 8)
    ):
    def plot(axes, plot_data, line_color, is_true, label):
        query_str = 'logreg' if is_true else 'not logreg'
        plot_data.query(query_str).plot(
            x='quarters_from_init',
            y='mean',
            ax=axes,
            label=label,
            color=line_color,
            yerr='CI',
            linewidth=3
        )
        first_point_y = plot_data.query(query_str + ' and quarters_from_init == 4')['mean'].iat[0]
        slope=first_point_y - plot_data.query(query_str + ' and quarters_from_init == 3')['mean'].iat[0]
        
        axs.plot(
            (4,forecast_length),
            (first_point_y,first_point_y+slope*forecast_length),
            linestyle='--',
            color=line_color,
            linewidth=3
        )
    data = rolling_outcomes.join(
        logreg_targets[logreg_key](filtered_data).rename('logreg'),
        how="inner"
    )
    data = data.groupby(['quarters_from_init', 'logreg'])[metric].agg(['mean', 'sem']).reset_index()
    data["CI"] = [norm.interval(confidence=0.95,loc=0,scale=sem[1])[1] for sem in data['sem'].items()]
    
    fig, axs = plt.subplots(figsize=fig_size)

    plot(axs,data,'black', True, true_label)
    plot(axs,data,'darkgrey', False, false_label)
    
    axs.set_xlabel('Quarters from Initial Visit')
    axs.set_ylabel(y_axis_label)
    axs.set_title(title)
    fig.savefig(f"../results/{analysis_name}/figures/comparison plots/{title}.jpg", bbox_inches='tight')

def plot_figures(
        analysis_name:str,
        chart_plotting_configurations:list[dict],
        logreg_targets:list[dict[str, callable]]
    ):
    filtered_data, rolling_outcomes = transform_data_to_quarterly_outcomes(analysis_name)
    for plot in chart_plotting_configurations:
        try:
            plot_chart(analysis_name, filtered_data, rolling_outcomes, logreg_targets, **plot)
        except Exception as e:
            print(f"Could not plot chart {plot['title']}")
            print(f"Exception encountered: {e}")
    plt.close('all')


