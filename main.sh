
module load deeplearning/2.3.0



### pretrain the model
    
python main.py \
    --pretrain \
    --jobname hgmd \
    --seq_files  /N/project/ENTEX/TL/data/HGMD.fasta \
    --snp_files  /N/project/ENTEX/TL/data/HGMD.bed \
    --frac_pretrain 1  \
    --frac_val 0.2  \
    --batch_size 128  \
    --epochs 50  
    
    

### compare four deep learning models

python main.py \
    --comparison\
    --jobname CAGI_train \
    --pretrain_model hgmd1.0.seq.h5  \
    --data_file /N/project/ENTEX/TL/data/CAGI_train.h5 \
    --frac_trains 0.1 1  \
    --frac_test 0.2  \
    --frac_val 0.2  \
    --methods tl self   \
    --nrep 5  \
    --batch_size 64  \
    --epochs 50  \
    --dropout 0.5  \
    --ilayer 2  \
    --nodes 128 64



    


