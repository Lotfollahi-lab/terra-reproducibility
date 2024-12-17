import os
import re
import yaml
import pandas as pd


tokenizer='cell-graph-tokenizer'
# tokenizer='cell-neighborhood-tokenizer'

# Read metadata csv and filter out non-xenium datasets
dataset_names_df = pd.read_csv('/lustre/scratch126/cellgen/team361/am84/NicheJEPA/reproducibility/analysis/tokenization/human_cohort1_50m_dataset_names.txt', index_col=0, header=None)

for dataset_name, _ in dataset_names_df.iterrows():
    print('Processing: ', dataset_name)
    dataset_id = re.search(r'(\d+)-', dataset_name).group(1)
    num_batches = re.search(r'-(\d+)b', dataset_name).group(1)

    # Build the data dictionary
    data_dict = {
        'experiment': {
            'experiment_name': dataset_name,
            'experiment_description': (
                f"{dataset_name} (dataset-id {dataset_id}) | {num_batches} batch | 1 panel"
            ),
            'tokenizer': tokenizer,
            'overwrite': False
        },
        'tokenization': {
            'token_dict_experiment_name': 'human_cohort1_50m',
            'tokens_per_cell': 128,
            'add_cls_tokens': False,
            'norm_method': "shifted_log",
            'norm_factor': None,
            'prenormalize': False,
            'num_shards': 32,
            'keep_in_memory': False,
            'n_neighs': 10,
            'radius': None,
            'delaunay': False
        },
        'wandb': {
            'project': "NicheJEPA",
            'job_type': "create-tokenized-dataset",
            'save_artifact': False,
            'mode': "offline"
        },
        'paths': {
            'data_directory_path': "/lustre/scratch126/cellgen/team361/DATASETS",
        },
    }

    # Define a custom Dumper to control formatting
    class MyDumper(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super(MyDumper, self).increase_indent(flow, False)

    # Dump the data to a YAML string
    yaml_str = yaml.dump(
        data_dict,
        Dumper=MyDumper,
        default_flow_style=False,
        sort_keys=False,
        width=1000
    )

    # Manually insert blank lines between top-level keys
    lines = yaml_str.split('\n')
    new_lines = []
    for i, line in enumerate(lines):
        if i > 0 and not line.startswith(' ') and lines[i - 1]:
            new_lines.append('')  # Insert blank line before top-level key
        new_lines.append(line)
    modified_yaml_str = '\n'.join(new_lines)

    # Write the modified YAML string to the file
    with open(f'config/{tokenizer}/{tokenizer}_{dataset_name}.yaml', 'w') as outfile:
        outfile.write(modified_yaml_str)