@ECHO OFF

:: %HOMEDRIVE% = C:
:: %HOMEPATH% = \Users\Ruben
:: %system32% ??
:: No spaces in paths
:: Program Files > ProgramFiles
:: cls = clear screen
:: CMD reads the system environment variables when it starts. To re-read those variables you need to restart CMD
:: Use console 2 http://sourceforge.net/projects/console/

SET HF_HOME="~/.cache/huggingface"
SET TRANSFORMERS_VERBOSITY="error"
SET UFF_SENTIMENT_HOME="~/uff-sentiment-analysis-ptbr"
SET UFF_SENTIMENT_EMBEDDINGS="~/.cache/embeddings"
SET UFF_CACHE_HOME="~/.cache"
SET WANDB_API_KEY="a3a73b0e8265efebf6707b23caff072f313a5c11"
SET WANDB_CACHE_DIR="~/.cache/wandb"
SET WANDB_SILENT=true
SET WANDB_START_METHOD="thread"

conda activate uff-sentiment-analysis

cd ~/repositories/github/sentiment-analysis-ptbr

cd ..
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install .

cd ~/repositories/github/sentiment-analysis-ptbr

#debug
#python run-static-word-embedding.py --datasets_index 0 1 --field 'tweet_normalized' --batch_size 2048 --classes 'multiclass' --selected_word_embeddings 'fasttext_bin'
#python run-contextual-word-embedding.py --datasets_index 0 --field 'tweet_normalized' --batch_size 256 --classes 'binary' --selected_transformers 'bertweetbr'
#python run-finetuning-downstream.py --train_epochs 1 --batch_size 16 --datasets_index 0 1 --field tweet_normalized --classes 'multiclass' --disable_tqdm --do evaluate --selected_transformers 'bertweetbr' --eval_type 'all'
#python run-finetuning-mlm.py --train_epochs 1 --batch_size 16 --datasets_index 0 1 --field 'tweet_normalized' --classes 'multiclass' --disable_tqdm --selected_transformers 'xlmr-base' --max_length 128 --strategy 'alldata'
#python run-from-finetuning-mlm.py --classes 'multiclass' --strategy 'loo' --experiment_type 'fine-tuning downstream' --batch_size 8 --disable_tqdm --results_dir 'fine-tuning downstream-ft' --log_dir 'fine-tuning downstream-ft' --selected_transformers 'xlmr-base'

################################################################################################# TRAINING CLASSIFIERS ####################################################################################################

#python3 run-static-word-embedding.py --datasets_index 0 1 2 3 4 5 6 7 --field 'tweet_normalized' --batch_size 2048 --classes 'binary' --selected_word_embeddings 'fasttext_bin' 'word2vec' 'glove' --save_classifiers --eval_type 'all' --do 'train'
#python3 run-static-word-embedding.py --datasets_index 0 1 2 3 4 --field 'tweet_normalized' --batch_size 2048 --classes 'multiclass' --selected_word_embeddings 'fasttext_bin' 'word2vec' 'glove' --save_classifiers --eval_type 'all' --do 'train'

#python3 run-contextual-word-embedding.py --datasets_index 0 1 2 3 4 5 6 7 --field 'tweet_normalized' --batch_size 512 --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --save_classifiers --eval_type 'all' --do 'train'
#python3 run-contextual-word-embedding.py --datasets_index 0 1 2 3 4 --field 'tweet_normalized' --batch_size 256 --classes 'multiclass' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --save_classifiers --eval_type 'all' --do 'train'

#python3 run-finetuning-downstream.py --train_epochs 3 --batch_size 32 --datasets_index 0 1 2 3 4 5 6 7 --field tweet_normalized --classes 'binary' --disable_tqdm --do evaluate --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --save_classifiers --eval_type 'all' --do 'train'
#python3 run-finetuning-downstream.py --train_epochs 3 --batch_size 32 --datasets_index 0 1 2 3 4 --field tweet_normalized --classes 'multiclass' --disable_tqdm --do evaluate --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --save_classifiers --eval_type 'all' --do 'train'

###################################################################### MODEL LANGUAGES AND WORD EMBEDDINGS AS FEATURE EXTRACTORS ##########################################################################################

#binary
#python run-static-word-embedding.py --datasets_index 0 1 2 --field 'tweet_normalized' --batch_size 2048 --classes 'binary' --selected_word_embeddings 'fasttext_bin' 'word2vec' 'glove'
#multiclass - reexecute because narrPT-final has been updated
#python run-static-word-embedding.py --datasets_index 0 1 2 3 4 --field 'tweet_normalized' --batch_size 2048 --classes 'multiclass' --selected_word_embeddings 'fasttext_bin' 'word2vec' 'glove'

#binary
#python run-contextual-word-embedding.py --datasets_index 0 1 2 --field 'tweet_normalized' --batch_size 512 --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'xlmt-base' 'bertweetbr'
#multiclass - reexecute because narrPT-final has been updated and to include xlmt and bertweetfr
python run-contextual-word-embedding.py --datasets_index 0 1 2 3 4 --field 'tweet_normalized' --batch_size 256 --classes 'multiclass' --selected_transformers 'xlmt-base'

######################################################################################### DOWNSTREAM TASK FINE-TUNING ######################################################################################################

#binary
#python run-finetuning-downstream.py --train_epochs 3 --batch_size 32 --datasets_index 0 1 2 --field tweet_normalized --classes 'binary' --disable_tqdm --do evaluate --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'xlmt-base' 'bertweet' 'bertweetbr' 'bertweetfr'
#multiclass
#python run-finetuning-downstream.py --train_epochs 3 --batch_size 32 --datasets_index 0 1 2 3 4 --field tweet_normalized --classes 'multiclass' --disable_tqdm --do evaluate --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'xlmt-base' 'bertweet' 'bertweetbr' 'bertweetfr'

############################################################################## FINE-TUNING MASKED LANGUAGE MODEL ON TWEETS-PT ###############################################################################################

#loo
#python run-finetuning-mlm.py --train_epochs 3 --batch_size 16 --datasets_index 0 1 2 3 4 5 6 7 --field tweet_normalized --classes 'binary' --disable_tqdm --selected_transformers 'xlmr-base' --max_length 128 --strategy 'loo'
#python run-finetuning-mlm.py --train_epochs 3 --batch_size 8 --datasets_index 0 1 2 3 4 --field tweet_normalized --classes 'multiclass' --disable_tqdm --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'loo'
#python run-finetuning-mlm.py --train_epochs 20 --batch_size 8 --datasets_index 0 1 2 3 4 5 6 7 --field tweet_normalized --classes 'mix' --disable_tqdm --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'xlmt-base' 'bertweet' 'bertweetbr' 'bertweetfr' --strategy 'loo' --local_files_only

#indata
#python run-finetuning-mlm.py --train_epochs 3 --batch_size 8 --datasets_index 0 1 2 3 4 5 6 7 --field tweet_normalized --classes 'binary' --disable_tqdm --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'indata'
#python run-finetuning-mlm.py --train_epochs 3 --batch_size 8 --datasets_index 0 1 2 3 4 --field tweet_normalized --classes 'multiclass' --disable_tqdm --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'indata'

#alldata
#python run-finetuning-mlm.py --train_epochs 3 --batch_size 8 --datasets_index 0 1 --field tweet_normalized --classes 'binary' --disable_tqdm --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'alldata'
#python run-finetuning-mlm.py --train_epochs 3 --batch_size 8 --datasets_index 0 1 2 3 4 --field tweet_normalized --classes 'multiclass' --disable_tqdm --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'alldata'

############################################################ FEATURE EXTRACTION EXPERIMENTS ON FINE-TUNING MASKED LANGUAGE MODEL ON TWEETS-PT ##############################################################################

#loo
#python run-from-finetuning-mlm.py --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'loo' --experiment_type 'feature extraction' --batch_size 256 --disable_tqdm
#python run-from-finetuning-mlm.py --classes 'multiclass' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'loo' --experiment_type 'feature extraction' --batch_size 256 --disable_tqdm
#python run-from-finetuning-mlm.py --classes 'mix' --selected_transformers 'xlmt-base' --strategy 'loo' --experiment_type 'feature extraction' --batch_size 256 --disable_tqdm --local_files_only --results_dir 'embeddings-ft' --log_dir 'embeddings-ft' 

#indata
#python run-from-finetuning-mlm.py --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'indata' --experiment_type 'feature extraction' --batch_size 256 --disable_tqdm
#python run-from-finetuning-mlm.py --classes 'multiclass' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'indata' --experiment_type 'feature extraction' --batch_size 256 --disable_tqdm

#alldata
#python run-from-finetuning-mlm.py --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'alldata' --experiment_type 'feature extraction' --batch_size 256 --disable_tqdm
#python run-from-finetuning-mlm.py --classes 'multiclass' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'alldata' --experiment_type 'feature extraction' --batch_size 256 --disable_tqdm

######################################################### DOWNSTREAM FINE-TUNING EXPERIMENTS ON FINE-TUNING MASKED LANGUAGE MODEL ON TWEETS-PT ##############################################################################

#loo
#python run-from-finetuning-mlm.py --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'loo' --experiment_type 'fine-tuning downstream' --batch_size 256 --disable_tqdm
#python run-from-finetuning-mlm.py --classes 'multiclass' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'loo' --experiment_type 'fine-tuning downstream' --batch_size 256 --disable_tqdm

#indata
#python run-from-finetuning-mlm.py --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'indata' --experiment_type 'fine-tuning downstream' --batch_size 256 --disable_tqdm
#python run-from-finetuning-mlm.py --classes 'multiclass' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'indata' --experiment_type 'fine-tuning downstream' --batch_size 256 --disable_tqdm

#alldata
#python run-from-finetuning-mlm.py --classes 'binary' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'alldata' --experiment_type 'fine-tuning downstream' --batch_size 256 --disable_tqdm
#python run-from-finetuning-mlm.py --classes 'multiclass' --selected_transformers 'mbert' 'bertimbau-base' 'xlmr-base' 'bertweet' 'bertweetbr' --strategy 'alldata' --experiment_type 'fine-tuning downstream' --batch_size 256 --disable_tqdm