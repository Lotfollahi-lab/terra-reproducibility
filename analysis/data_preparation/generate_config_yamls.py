import pandas as pd
import yaml
import os

# Read metadata csv and filter out non-xenium datasets
meta0 = pd.read_csv('/lustre/scratch126/cellgen/team361/bc8/cytoweave/nichejepa/actions/nichejepa_updated_meta.csv', index_col=0)
meta0 = meta0.drop(columns=['data_access_link', 'raw_on_farm', 'H&E_on_farm'])
meta0 = meta0.loc[~meta0['dataset_id'].isna()]
meta0 = meta0.loc[~meta0['source'].isna()]
meta0 = meta0.loc[:,["ontology" not in i for i in meta0.columns]]
meta0['dataset_id'] = [int(i) for i in meta0['dataset_id']]
did_counts = meta0['dataset_id'].value_counts()

for dataset_name, row in meta0.iterrows():
    # Generate a short code from uppercase letters in the sample name
    ds_name = row['sample'].split('-')[0]
    # Remove numbers from ds_name
    ds_name = ''.join(c for c in ds_name if not c.isdigit())
    ds_short = ''.join(c[0].lower() for c in ds_name.split())

    # Construct the experiment name and extract dataset_id & total number of batches
    dataset_id = row['dataset_id']
    total_batch = did_counts.get(dataset_id, 0)  # Use .get() to avoid KeyError
    exp_name = f"{ds_short[:3]}{dataset_id}-{total_batch}b_1p"

    # Determine sample names and URLs based on the total_batch
    if total_batch > 1:
        meta_subset = meta0[meta0['dataset_id'] == dataset_id]
        sample_names = meta_subset.index.tolist()
        urls = meta_subset['download_url'].tolist()
    else:
        sample_names = [dataset_name]
        urls = [row['download_url']]

    if row['source'] == 'offline':
        urls = []

    # Build the data dictionary
    data_dict = {
        'experiment': {
            'name': exp_name,
            'description': (
                f"{ds_name}(dataset-id {dataset_id}) | {total_batch} batch | 1 panel | "
                "Download from 10x Genomics to Bronze and preprocess to silver"
            ),
            'study': row['study']
        },
        'paths': {
            'data_directory_path': "/lustre/scratch126/cellgen/team361/DATASETS",
        },
        'dataset': {
            'overwrite': False,
            'source': row['source'],
            'download_params': {
                'redownload': False,
                'urls': urls,
            },
            'file_names': sample_names
        },
        'harmonization': {
            'uns': {
                'dataset_id': dataset_id,
                'assay': row['assay'],
                'species': row['species'],
                'tissue': row['tissue'],
            },
            'batch': {
                'split_by_batch': False,
                'sample_batch_map': {},
            },
            'key_mappings': {},
            'misc': {
                'min_genes': 10,
            }
        }
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
    with open(f'config/bronze_to_silver/{exp_name}.yaml', 'w') as outfile:
        outfile.write(modified_yaml_str)
