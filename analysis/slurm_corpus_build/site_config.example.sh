# Site settings of the Slurm corpus build (submit_corpus_build.sh --config <file>).
# Copy this file, edit it, and keep the copy outside the repository: it holds
# site paths.

# Name of the corpus: folder and file prefix of its norm factors
# ($TERRA_DATA_DIR/tokenizer/$CORPUS_NAME/${CORPUS_NAME}_norm_factors.csv)
export CORPUS_NAME=my_corpus

# Data root. The build writes silver/<dataset>/, gold/cell-graph-tokenizer/<dataset>/,
# tokenizer/$CORPUS_NAME/, work/, logs/ and cache/ (the Hugging Face datasets
# cache of the tokenizer, about as large as the tokenized parts; delete it
# once every tokenization job has finished) here.
export TERRA_DATA_DIR=/path/to/terra_data

# Shell command(s) that activate a Python environment with TERRA (branch
# feat/compact-tokenized-data) and wandb, for example
#   'source /path/to/venv/bin/activate'
#   'module load miniforge3 && eval "$(conda shell.bash hook)" && conda activate terra'
export TERRA_ENV_ACTIVATE='source /path/to/venv/bin/activate'

# Token dictionary (.pkl); its Ensembl IDs are the gene vocabulary
export TOKEN_DICTIONARY=$TERRA_DATA_DIR/tokenizer/token_dictionary_homo_sapiens.pkl
# Gene name -> Ensembl ID map (.pkl) of the corpus harmonization (TERRA's
# harmonize_adata maps new data with the same kind of map). Optional: without
# it, var needs a column of Ensembl IDs (e.g. 10x 'gene_ids').
export GENE_NAME_MAP=$TERRA_DATA_DIR/genes/homo_sapiens_gene_name_to_ensembl_id_dict.pkl

# Optional: samples left out of the norm factors, e.g. validation / test
# samples (one '<dataset name> <sample>' pair per line; a third column is
# ignored, so the SPLIT_SAMPLES file can be given). Empty: all cells.
export HELDOUT_SAMPLES=

# Splits (stage split): validation and test samples, one
# '<dataset name> <sample> <val|test>' per line; corpus_tools.py
# propose-split-samples writes a random proposal to edit. Empty: no split stage.
export SPLIT_SAMPLES=
# Split variants, one model each: 'name=dataset,dataset name=dataset ...'.
# Empty: one variant 'all' with every dataset of the table.
export SPLITS=""

# Slurm
export SLURM_PARTITION=cpu
export SLURM_EXTRA_ARGS=""   # e.g. "--account=myproject --qos=normal"

# Resources per stage (the defaults)
export INSPECT_CPUS=2 INSPECT_MEM=64G INSPECT_TIME=02:00:00
export PREP_CPUS=4 PREP_MEM=256G PREP_TIME=12:00:00
export NORM_CPUS=2 NORM_MEM=128G NORM_TIME=24:00:00
export TOKENIZE_CPUS=16 TOKENIZE_MEM=256G TOKENIZE_TIME=48:00:00
export CONCAT_CPUS=8 CONCAT_MEM=64G CONCAT_TIME=24:00:00
export SPLIT_CPUS=2 SPLIT_MEM=32G SPLIT_TIME=06:00:00
# Tokenization processes: by default as many of TOKENIZE_CPUS as the memory
# estimate fits into TOKENIZE_MEM (corpus_tools.py nproc); set to override
#export TOKENIZE_NPROC=8
