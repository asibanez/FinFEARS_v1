INPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/04_spy_project_FEARS/00_data/01_preprocessed
OUTPUT_DIR=C:/Users/siban/Dropbox/BICTOP/MyInvestor/06_model/02_NLP/04_spy_project_FEARS/00_data/02_runs/99_TEST_DELETE

#INPUT_DIR=/home/sibanez/Projects/01_MyInvestor_FEARS/00_data/01_preprocessed
#OUTPUT_DIR=/home/sibanez/Projects/01_MyInvestor_FEARS/00_data/02_runs/99_TEST_DELETE

python -m ipdb train_test.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --task=Train \
    \
    --model_name=ProsusAI/finbert \
    --seq_len=256 \
    --hidden_dim=768 \
    --alpha=0.5 \
    --att_dim=128 \
    --freeze_BERT=True \
    --seed=1234 \
    --use_cuda=True \
    \
    --n_epochs=10 \
    --batch_size_train=1000 \
    --shuffle_train=False \
    --drop_last_train=False \
    --dev_train_ratio=1 \
    --train_toy_data=False \
    --len_train_toy_data=30 \
    --lr=2e-6 \
    --wd=1e-6 \
    --dropout=0.2 \
    --momentum=0.9 \
    --save_final_model=True \
    --save_model_steps=True \
    --save_step_cliff=9 \
    --gpu_ids_train=0 \
    \
    --test_file=model_test.pkl \
    --model_file=model.pt.9 \
    --batch_size_test=1000 \
    --gpu_id_test=0 \

read -p 'EOF'

#--model_name=nlpaueb/legal-bert-small-uncased \
#--hidden_dim=512 \

#--task=Train / Test
#--batch_size=280 / 0,1,2,3
