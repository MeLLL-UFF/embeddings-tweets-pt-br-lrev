#!/bin/bash
#SBATCH --job-name=contextual # Job name
#SBATCH --partition=gpu							                # Partition Type
#SBATCH --ntasks=12                   
#SBATCH --mem=250gb
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fpcar@yahoo.com.br     
#SBATCH --output=logs/sentiment-analysis-ptbr_%j.out			# Path to the standard output file relative to the working directory
#SBATCH --error=logs/sentiment-analysis-ptbr_%j.err				# Path to the standard error file relative to the working directory
#SBATCH --nodelist=cds1gpun01

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"

export HF_HOME="/home/Y435/.cache/huggingface"
export TRANSFORMERS_VERBOSITY="error"
export UFF_SENTIMENT_HOME="/home/Y435/uff-sentiment-analysis-ptbr"
export UFF_SENTIMENT_EMBEDDINGS="/home/Y435/.cache/embeddings"
export UFF_CACHE_HOME="/home/Y435/.cache"
export PATH="~/miniconda3/bin":$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniconda3/lib
export WANDB_API_KEY="a3a73b0e8265efebf6707b23caff072f313a5c11"
export WANDB_CACHE_DIR="~/.cache/wandb"
export WANDB_SILENT=true
export WANDB_START_METHOD="thread"

export https_proxy=http://user:pwd@url:port
export http_proxy=http://user:pwd@url:port

cd ~/repositories/github/sentiment-analysis-ptbr
conda init bash
conda env create -f uff-sentiment-analysis.yml
conda activate uff-sentiment-analysis
cd ..
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install .
cd ~/repositories/github/sentiment-analysis-ptbr

#debug
#python3 run-static-word-embedding.py --datasets_index 0 1 --field 'tweet_normalized' --batch_size 2048 --classes 'multiclass' --selected_word_embeddings 'fasttext_bin'
#python3 run-contextual-word-embedding.py --datasets_index 0 --field 'tweet_normalized' --batch_size 256 --classes 'binary' --selected_transformers 'bertweetbr'
#python3 run-finetuning-downstream.py --train_epochs 1 --batch_size 16 --datasets_index 0 --field tweet_normalized --classes 'multiclass' --disable_tqdm --do evaluate --selected_transformers 'bertweetbr'
#python3 run-finetuning-mlm.py --train_epochs 1 --batch_size 16 --datasets_index 0 1 --field 'tweet_normalized' --classes 'multiclass' --disable_tqdm --selected_transformers 'xlmr-base' --max_length 128 --strategy 'alldata'
#python3 run-from-finetuning-mlm.py --classes 'multiclass' --strategy 'alldata' --experiment_type 'fine-tuning downstream' --batch_size 8 --disable_tqdm --results_dir 'fine-tuning downstream-ft' --log_dir 'fine-tuning downstream-ft' --selected_transformers 'xlmr-base'
#python3 run-from-finetuning-mlm.py --classes 'binary' --selected_transformers 'bertweetbr' --strategy 'loo' --experiment_type 'feature extraction' --batch_size 256 --disable_tqdm

################################################################################################# TRAINING CLASSIFIERS ####################################################################################################

python3 run-static-word-embedding.py --datasets_index 0 1 2 3 4 5 6 7 --field 'tweet_normalized' --batch_size 2048 --classes 'binary' --selected_word_embeddings 'fasttext_bin' 'word2vec' 'glove' --save_classifiers --eval_type 'all' --do 'train'
python3 run-static-word-embedding.py --datasets_index 0 1 2 3 4 5 6 7 --field 'tweet_normalized' --batch_size 2048 --classes 'multiclass' --selected_word_embeddings 'fasttext_bin' 'word2vec' 'glove' --save_classifiers --eval_type 'all' --do 'train'
python3 run-contextual-word-embedding.py --datasets_index 0 1 2 3 4 5 6 7 --field 'tweet_normalized' --batch_size 512 --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --save_classifiers --eval_type 'all' --do 'train'
python3 run-contextual-word-embedding.py --datasets_index 0 1 2 3 4 --field 'tweet_normalized' --batch_size 256 --classes 'multiclass' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --save_classifiers --eval_type 'all' --do 'train'

python3 run-finetuning-downstream.py --train_epochs 3 --batch_size 32 --datasets_index 0 1 2 3 4 5 6 7 --field tweet_normalized --classes 'binary' --disable_tqdm --do evaluate --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --save_classifiers --eval_type 'all' --do 'train'
python3 run-finetuning-downstream.py --train_epochs 3 --batch_size 32 --datasets_index 0 1 2 3 4 --field tweet_normalized --classes 'multiclass' --disable_tqdm --do evaluate --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --save_classifiers --eval_type 'all' --do 'train'

###################################################################### MODEL LANGUAGES AND WORD EMBEDDINGS AS FEATURE EXTRACTORS ##########################################################################################

#binary
#python3 run-static-word-embedding.py --datasets_index 0 1 2 --field 'tweet_normalized' --batch_size 2048 --classes 'binary' --selected_word_embeddings 'fasttext_bin' 'word2vec' 'glove'
#multiclass
#python3 run-static-word-embedding.py --datasets_index 0 1 2 3 4 --field 'tweet_normalized' --batch_size 2048 --classes 'multiclass' --selected_word_embeddings 'fasttext_bin' 'word2vec' 'glove'

#binary
#python3 run-contextual-word-embedding.py --datasets_index 0 1 2 --field 'tweet_normalized' --batch_size 512 --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'xlmt-base' 'bertweetbr'
#multiclass
#python3 run-contextual-word-embedding.py --datasets_index 0 1 2 3 4 --field 'tweet_normalized' --batch_size 256 --classes 'multiclass' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'xlmt-base' 'bertweet' 'bertweetbr' 'bertweetfr'

######################################################################################### DOWNSTREAM TASK FINE-TUNING ######################################################################################################

#binary
#python3 run-finetuning-downstream.py --train_epochs 3 --batch_size 32 --datasets_index 0 1 2 --field tweet_normalized --classes 'binary' --disable_tqdm --do evaluate --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'xlmt-base' 'bertweet' 'bertweetbr' 'bertweetfr'
#multiclass
#python3 run-finetuning-downstream.py --train_epochs 3 --batch_size 32 --datasets_index 0 1 2 3 4 --field tweet_normalized --classes 'multiclass' --disable_tqdm --do evaluate --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'xlmt-base' 'bertweet' 'bertweetbr' 'bertweetfr'

############################################################################## FINE-TUNING MASKED LANGUAGE MODEL ON TWEETS-PT ###############################################################################################

#loo
#python3 run-finetuning-mlm.py --train_epochs 3 --batch_size 16 --datasets_index 0 1 2 3 4 5 6 7 --field tweet_normalized --classes 'binary' --disable_tqdm --selected_transformers 'xlmr-base' --max_length 128 --strategy 'loo'
#python3 run-finetuning-mlm.py --train_epochs 3 --batch_size 8 --datasets_index 0 1 2 3 4 --field tweet_normalized --classes 'multiclass' --disable_tqdm --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'loo'
#python run-finetuning-mlm.py --train_epochs 20 --batch_size 8 --datasets_index 0 1 2 3 4 5 6 7 --field tweet_normalized --classes 'mix' --disable_tqdm --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'xlmt-base' 'bertweet' 'bertweetbr' 'bertweetfr' --strategy 'loo'

#indata
#python3 run-finetuning-mlm.py --train_epochs 3 --batch_size 8 --datasets_index 0 1 2 3 4 5 6 7 --field tweet_normalized --classes 'binary' --disable_tqdm --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'indata'
#python3 run-finetuning-mlm.py --train_epochs 3 --batch_size 8 --datasets_index 0 1 2 3 4 --field tweet_normalized --classes 'multiclass' --disable_tqdm --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'indata'

#alldata
#python3 run-finetuning-mlm.py --train_epochs 3 --batch_size 8 --datasets_index 0 1 --field tweet_normalized --classes 'binary' --disable_tqdm --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'alldata'
#python3 run-finetuning-mlm.py --train_epochs 3 --batch_size 8 --datasets_index 0 1 2 3 4 --field tweet_normalized --classes 'multiclass' --disable_tqdm --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'alldata'

############################################################ FEATURE EXTRACTION EXPERIMENTS ON FINE-TUNING MASKED LANGUAGE MODEL ON TWEETS-PT ##############################################################################

#loo
#python3 run-from-finetuning-mlm.py --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'loo' --experiment_type 'feature extraction' --batch_size 256 --disable_tqdm
#python3 run-from-finetuning-mlm.py --classes 'multiclass' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'loo' --experiment_type 'feature extraction' --batch_size 256 --disable_tqdm

#indata
#python3 run-from-finetuning-mlm.py --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'indata' --experiment_type 'feature extraction' --batch_size 256 --disable_tqdm
#python3 run-from-finetuning-mlm.py --classes 'multiclass' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'indata' --experiment_type 'feature extraction' --batch_size 256 --disable_tqdm

#alldata
#python3 run-from-finetuning-mlm.py --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'alldata' --experiment_type 'feature extraction' --batch_size 256 --disable_tqdm
#python3 run-from-finetuning-mlm.py --classes 'multiclass' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'alldata' --experiment_type 'feature extraction' --batch_size 256 --disable_tqdm


######################################################### DOWNSTREAM FINE-TUNING EXPERIMENTS ON FINE-TUNING MASKED LANGUAGE MODEL ON TWEETS-PT ##############################################################################

#loo
#python3 run-from-finetuning-mlm.py --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'loo' --experiment_type 'fine-tuning downstream' --batch_size 256 --disable_tqdm
#python3 run-from-finetuning-mlm.py --classes 'multiclass' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'loo' --experiment_type 'fine-tuning downstream' --batch_size 256 --disable_tqdm

#indata
#python3 run-from-finetuning-mlm.py --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'indata' --experiment_type 'fine-tuning downstream' --batch_size 256 --disable_tqdm
#python3 run-from-finetuning-mlm.py --classes 'multiclass' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'indata' --experiment_type 'fine-tuning downstream' --batch_size 256 --disable_tqdm

#alldata
#python3 run-from-finetuning-mlm.py --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'alldata' --experiment_type 'fine-tuning downstream' --batch_size 256 --disable_tqdm
#python3 run-from-finetuning-mlm.py --classes 'multiclass' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'alldata' --experiment_type 'fine-tuning downstream' --batch_size 256 --disable_tqdm