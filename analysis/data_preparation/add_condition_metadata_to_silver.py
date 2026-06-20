import os
import yaml
import scanpy as sc

dataset_ids = [
'xhb23-1b_1p',
'xhb24-2b_1p',
'xhb25-2b_1p',
'xhb26-2b_1p',
'xhb27-1b_1p',
'xhl28-1b_1p',
'xhl29-1b_1p',
'xhp30-1b_1p',
'xhs31-2b_1p',
'xhh32-1b_1p',
'xhln33-1b_1p',
'xhp34-1b_1p',
'xhp35-1b_1p',
'xhs36-2b_1p',
'xhb37-3b_1p',
'xhc38-4b_1p',
'xhk39-2b_1p',
'xhl40-2b_2p',
'xhl41-2b_1p',
'xhb42-3b_1p',
'xht43-2b_1p',
'xhsc44-6b_1p',
'xhc45-1b_1p',
'xhl46-1b_1p',
'xhp48-1b_1p',
'xhs49-1b_1p',
'xho50-1b_1p',
'xhc51-1b_1p',
'xhd52-1b_1p',
'xhl53-1b_1p',
'xho54-1b_1p',
'xhc55-3b_1p',
'xhs62-1b_1p',
'xhb76-16b_1p',
'xhk79-4b_1p',
'xhs1000-39b_1p',
'xhb1002-AT3-1b_1p',
'xhb1002-AT4-3b_1p',
'xhb1002-AT5-1b_1p',
'xhb1002-AT9-6b_1p',
'xhb1002-AT10-7b_1p',
'xhb1002-AT12-1b_1p',
'xhb1002-AT13-2b_1p',
'xhb1002-AT14-1b_1p',
'xhb1002-AT15-10b_1p',
'xhb1002-AT16-15b_1p',
'xhb1002-AT17-2b_1p',
'xhb1002-AT18-2b_1p',
'xhb1002-AT19-1b_1p',
'xhb1002-AT20-2b_1p',
'xhb1002-AT21-2b_1p',
'xhb1002-AT23-9b_1p',
'xhb1002-AT101-6b_1p',
'xhb1003-1b_1p',
'xhln1004-1b_1p',
'xhl1005-1b_1p',
'mhb14-1b_1p',
'mhc15-2b_1p',
'mhl16-2b_1p',
'mhl17-2b_1p',
'mhm18-2b_1p',
'mho19-4b_1p',
'mhp20-2b_1p',
'mhu21-3b_1p',
'mhl63-10b_1p',
'chb56-1b_1p',
'chl57-2b_1p',
'chl59-8b_1p',
'chc61-9b_1p',
'chr78-11b_1p',
'chp60-1b_1p',
'mhh22-3b_1p',
'shp75-4b_1p',
'xhb1002-AT6-1-1b_1p',
'xhb1002-AT6-2-1b_1p',
'xhb1002-AT6-3-1b_1p',
'xhb1002-AT6-4-1b_1p',
'xho47-S1-1b_1p',
'xho47-S2-1b_1p',
'xhsc1006-2b_1p',
'xhc1001-1b_1p',
'mhl80-8b_1p',
'xhb81-1b_1p',
'xhl82-2b_1p',
'mhb83-1b_1p',
'xhk1007-CV1-41b_1p',
'xhk1007-CV2-41b_1p',
'xhk1007-CV3-41b_1p',
'xhk1007-CV4-41b_1p',
'xhk1007-CV5-41b_1p',
'xhk1007-CV6-41b_1p',
'xhk1007-CV7-41b_1p',
'xhk1007-CV8-41b_1p',
'xhk1007-CV9-41b_1p',
'xhe1008-1-20b_1p',
'xhe1008-2-20b_1p',
'xhe1008-3-20b_1p',
'xhe1008-4-20b_1p',
'xhe1008-5-20b_1p',
'xhe1008-6-20b_1p',
'xhe1008-7-20b_1p',
'xhe1008-8-20b_1p',
'xhe1008-9-20b_1p',
'xhe1008-10-20b_1p',
'xhs1009-17b_1p',
'xhs1010-27b_1p',
'xhs1011-22b_1p',
'xho1012-AU14-6b_1p',
'xho1012-AU16-7b_1p',
'xho1012-AU18-6b_1p',
'xho1012-AU20-6b_1p',
'xho1012-AU21-7b_1p',
'xho1012-AU22-6b_1p',
'xho1012-AU30-6b_1p',
'xho1012-AUrv15-6b_1p',
'xho1012-Hrv66-2b_1p',
'xho1012-Hrv69-4b_1p',
'xho1012-Hrv162-2b_1p',
'xho1012-Hrv182-4b_1p',
'xho1012-M23-4b_1p',
'xhl1013-4b_1p']

config_folder_path = '/nfs/team361/sb75/terra-reproducibility/config/bronze_to_silver'
silver_folder_path = '/nfs/team361/sb75/DATASETS/silver'

yaml_data = []

# Loop through the directory
for config_file_name in os.listdir(config_folder_path):
    
    if (config_file_name.endswith(".yaml") or config_file_name.endswith(".yml")) and (config_file_name.split('.')[0] in dataset_ids):
        config_file_path = os.path.join(config_folder_path, config_file_name)
        with open(config_file_path, 'r') as f:
            try:
                data = yaml.safe_load(f)
                yaml_data.append(data)
            except yaml.YAMLError as exc:
                print(f"Error parsing {filename}: {exc}")

        try:
    
            if 'condition' in data['harmonization']['uns'].keys():
                condition = data['harmonization']['uns']['condition']
    
                adata_silver_folder_path = f'{silver_folder_path}/{config_file_name.split(".")[0]}'
                for subdir, _, files in os.walk(adata_silver_folder_path):
                    for file_idx, file in enumerate(files):
                        if file.endswith('.h5ad'):
                            adata_file_path = os.path.join(subdir, file)
                            
                            adata = sc.read_h5ad(adata_file_path)
                            adata.uns['condition'] = condition
                            adata.write(adata_file_path)
                
            else:
    
                adata_silver_folder_path = f'{silver_folder_path}/{config_file_name.split(".")[0]}'
                for subdir, _, files in os.walk(adata_silver_folder_path):
                    for file_idx, file in enumerate(files):
                        if file.endswith('.h5ad'):
                            adata_file_path = os.path.join(subdir, file)
    
                            if file.split(".")[0] in data['harmonization']['uns']['sample_condition_map'].keys():
                                condition = data['harmonization']['uns']['sample_condition_map'][file.split(".")[0]]
                                adata = sc.read_h5ad(adata_file_path)
                                adata.uns['condition'] = condition
                                adata.write(adata_file_path)
                            else:
                                adata = sc.read_h5ad(adata_file_path)
                                sample = adata.obs[data['harmonization']['batch']['batch_key']].values.tolist()[0]
                                condition = data['harmonization']['uns']['sample_condition_map'][sample]
                                adata.uns['condition'] = condition
                                adata.write(adata_file_path)

            print('=====')
            print("SUCCESS")
            print(file)
            print(adata_silver_folder_path)
            print(condition)
            print('=====')

        except Exception as e:
            print('=====')
            print(f"An error occurred: {e}")
            print("FAILURE")
            print(file)
            print(adata_silver_folder_path)
            print(condition)
            print('=====')
            

# Now `yaml_data` is a list of all the loaded YAML contents
print(f"Loaded {len(yaml_data)} YAML files.")

for folder_name in [
'xhb1002-AT3-1b_1p',
'xhb1002-AT4-3b_1p',
'xhb1002-AT5-1b_1p',
'xhb1002-AT9-6b_1p',
'xhb1002-AT10-7b_1p',
'xhb1002-AT12-1b_1p',
'xhb1002-AT13-2b_1p',
'xhb1002-AT14-1b_1p',
'xhb1002-AT15-10b_1p',
'xhb1002-AT16-15b_1p',
'xhb1002-AT17-2b_1p',
'xhb1002-AT18-2b_1p',
'xhb1002-AT19-1b_1p',
'xhb1002-AT20-2b_1p',
'xhb1002-AT21-2b_1p',
'xhb1002-AT23-9b_1p',
'xhb1002-AT101-6b_1p']:
    adata_silver_folder_path = f'{silver_folder_path}/{folder_name}'
    for subdir, _, files in os.walk(adata_silver_folder_path):
        for file_idx, file in enumerate(files):
            if file.endswith('.h5ad'):
                adata_file_path = os.path.join(subdir, file)
                adata = sc.read_h5ad(adata_file_path)
                adata.uns['condition'] = 'glioblastoma'
                adata.write(adata_file_path)

for folder_name in [
'xhb1002-AT6-1-1b_1p',
'xhb1002-AT6-2-1b_1p',
'xhb1002-AT6-3-1b_1p',
'xhb1002-AT6-4-1b_1p']:
    adata_silver_folder_path = f'{silver_folder_path}/{folder_name}'
    for subdir, _, files in os.walk(adata_silver_folder_path):
        for file_idx, file in enumerate(files):
            if file.endswith('.h5ad'):
                adata_file_path = os.path.join(subdir, file)
                adata = sc.read_h5ad(adata_file_path)
                adata.uns['condition'] = 'glioblastoma'
                adata.write(adata_file_path)

for folder_name in [
'xhe1008-1-20b_1p',
'xhe1008-2-20b_1p',
'xhe1008-3-20b_1p',
'xhe1008-4-20b_1p',
'xhe1008-5-20b_1p',
'xhe1008-6-20b_1p',
'xhe1008-7-20b_1p',
'xhe1008-8-20b_1p',
'xhe1008-9-20b_1p',
'xhe1008-10-20b_1p']:
    adata_silver_folder_path = f'{silver_folder_path}/{folder_name}'
    for subdir, _, files in os.walk(adata_silver_folder_path):
        for file_idx, file in enumerate(files):
            if file.endswith('.h5ad'):
                adata_file_path = os.path.join(subdir, file)
                adata = sc.read_h5ad(adata_file_path)
                adata.uns['condition'] = 'healthy'
                adata.write(adata_file_path)

for folder_name in [
'xho47-S1-1b_1p',
'xho47-S2-1b_1p']:
    adata_silver_folder_path = f'{silver_folder_path}/{folder_name}'
    for subdir, _, files in os.walk(adata_silver_folder_path):
        for file_idx, file in enumerate(files):
            if file.endswith('.h5ad'):
                adata_file_path = os.path.join(subdir, file)
                adata = sc.read_h5ad(adata_file_path)
                adata.uns['condition'] = 'adenocarcinoma'
                adata.write(adata_file_path)

for folder_name in [
'xho1012-AU14-6b_1p',
'xho1012-AU16-7b_1p',
'xho1012-AU18-6b_1p',
'xho1012-AU20-6b_1p',
'xho1012-AU21-7b_1p',
'xho1012-AU22-6b_1p',
'xho1012-AU30-6b_1p',
'xho1012-AUrv15-6b_1p',
'xho1012-Hrv66-2b_1p',
'xho1012-Hrv69-4b_1p',
'xho1012-Hrv162-2b_1p',
'xho1012-Hrv182-4b_1p',
'xho1012-M23-4b_1p']:
    adata_silver_folder_path = f'{silver_folder_path}/{folder_name}'
    for subdir, _, files in os.walk(adata_silver_folder_path):
        for file_idx, file in enumerate(files):
            if file.endswith('.h5ad'):
                adata_file_path = os.path.join(subdir, file)
                adata = sc.read_h5ad(adata_file_path)
                adata.uns['condition'] = 'healthy'
                adata.write(adata_file_path)

sample_condition_map = {
  'CV1-KID-0-FO-1': 'renal_cell_carcinoma',
  'CV1-KID-0-FT-1': 'renal_cell_carcinoma',
  'CV1-KID-0-FT-2': 'renal_cell_carcinoma',
  'CV1-KID-0-FT-3': 'renal_cell_carcinoma',
  'CV2-KID-0-FO-1': 'renal_cell_carcinoma',
  'CV2-KID-0-FO-2': 'renal_cell_carcinoma',
  'CV2-KID-0-FT-1-s2': 'renal_cell_carcinoma',
  'CV2-KID-0-FT-2': 'renal_cell_carcinoma',
  'CV3-KID-0-FT-1-s3': 'renal_cell_carcinoma',
  'CV3-KID-0-FT-2-s3': 'renal_cell_carcinoma',
  'CV3-KID-0-FT-3-s3': 'renal_cell_carcinoma',
  'CV3-KID-0-FT-4-s3': 'renal_cell_carcinoma',
  'CV3-KID-0-FT-5': 'renal_cell_carcinoma',
  'CV4-KID-0-FO-1': 'renal_cell_carcinoma',
  'CV4-KID-0-FT-1-s3': 'renal_cell_carcinoma',
  'CV4-KID-0-FT-2-s3': 'renal_cell_carcinoma',
  'CV4-KID-0-FT-3-S3': 'renal_cell_carcinoma',
  'CV4-KID-0-FT-4-S3': 'renal_cell_carcinoma',
  'CV5-KID-0-FO-1': 'renal_cell_carcinoma',
  'CV5-KID-0-FT-1-s3': 'renal_cell_carcinoma',
  'CV5-KID-0-FT-2-s3': 'renal_cell_carcinoma',
  'CV5-KID-0-FT-4-S3': 'renal_cell_carcinoma',
  'CV6-KID-0-FO-1': 'renal_cell_carcinoma',
  'CV6-KID-0-FT-1': 'renal_cell_carcinoma',
  'CV6-KID-0-FT-2-s3': 'renal_cell_carcinoma',
  'CV6-KID-0-FT-3-s3': 'renal_cell_carcinoma',
  'CV6-KID-0-FT-4-S3': 'renal_cell_carcinoma',
  'CV6-KID-0-FT-5-S3': 'healthy',
  'CV7-KID-0-FO-1': 'renal_cell_carcinoma',
  'CV7-KID-0-FT-1-s3': 'renal_cell_carcinoma',
  'CV7-KID-0-FT-2-s3': 'renal_cell_carcinoma',
  'CV7-KID-0-FT-3-S3': 'renal_cell_carcinoma',
  'CV7-KID-0-FT-4-S3': 'renal_cell_carcinoma',
  'CV8-KID-0-FO-1': 'renal_cell_carcinoma',
  'CV8-KID-0-FT-1-s3': 'renal_cell_carcinoma',
  'CV9-KID-0-FO-1': 'renal_cell_carcinoma',
  'CV9-KID-0-FT-1-s3': 'renal_cell_carcinoma',
  'CV9-KID-0-FT-2-S3': 'renal_cell_carcinoma',
  'CV9-KID-0-FT-3-S3': 'renal_cell_carcinoma',
  'CV9-KID-0-FT-4-S3': 'renal_cell_carcinoma',
  'CV9-KID-0-FT-5-S4': 'healthy',
}

for folder_name in [
'xhk1007-CV1-41b_1p',
'xhk1007-CV2-41b_1p',
'xhk1007-CV3-41b_1p',
'xhk1007-CV4-41b_1p',
'xhk1007-CV5-41b_1p',
'xhk1007-CV6-41b_1p',
'xhk1007-CV7-41b_1p',
'xhk1007-CV8-41b_1p',
'xhk1007-CV9-41b_1p']:
    adata_silver_folder_path = f'{silver_folder_path}/{folder_name}'
    for subdir, _, files in os.walk(adata_silver_folder_path):
        for file_idx, file in enumerate(files):
            if file.endswith('.h5ad'):
                adata_file_path = os.path.join(subdir, file)
                adata = sc.read_h5ad(adata_file_path)

                sample = adata.obs['batch'].values.tolist()[0]
                condition = sample_condition_map[sample]
                adata.uns['condition'] = condition

                adata.write(adata_file_path)