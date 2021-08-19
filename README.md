# TLVar
Exploiting deep transfer learning for the prediction of functional noncoding variants using DNA sequence

## Introduction
We will propose a deep transfer learning model, which is based on deep convolutional neural network, to improve the prediction for functional noncoding variants from a specific context. To overcome the challenge of few validated functional noncoding variants, the transfer learning model consists of pretrained layers trained by large-scale generic functional noncoding variants, and retrained layers by context-specific functional noncoding variants with the pretrained layers frozen.


## Requirements and Installation

TLVar is implemented by TensorFlow/Keras.

- Python 3.8
- Keras == 2.4.0
- TensorFlow == 2.3.0
- numpy >= 1.15.4
- scipy >= 1.2.1
- scikit-learn >= 0.20.3
- seaborn >=0.9.0
- matplotlib >=3.1.0


Download MDeep:
```
git clone https://github.com/lichen-lab/TLVar
```


```

python main.py -h

usage: main.py [-h] [--comparison] [--pretrain] [--jobname JOBNAME]
               [--pretrain_model PRETRAIN_MODEL] [--data_file DATA_FILE]
               [--seq_file SEQ_FILE] [--snp_file SNP_FILE]
               [--seq_files SEQ_FILES [SEQ_FILES ...]]
               [--snp_files SNP_FILES [SNP_FILES ...]]
               [--frac_pretrain FRAC_PRETRAIN]
               [--frac_trains FRAC_TRAINS [FRAC_TRAINS ...]]
               [--frac_test FRAC_TEST] [--frac_val FRAC_VAL]
               [--methods METHODS [METHODS ...]] [--nrep NREP]
               [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--dropout DROPOUT]
               [--ilayer ILAYER] [--nodes NODES [NODES ...]]


optional arguments:
  -h, --help            show this help message and exit
  --comparison          Use this option for comparing TL model
  --pretrain            Use this option for pretrain model
  --jobname JOBNAME     Job name for model, figure and result
  --pretrain_model PRETRAIN_MODEL
                        Pretrained model h5 file
  --data_file DATA_FILE
                        h5 data file for x y
  --seq_file SEQ_FILE   fasta sequence for SNPs
  --snp_file SNP_FILE   label for SNPs
  --seq_files SEQ_FILES [SEQ_FILES ...]
                        fasta sequence for SNPs for multiple diseases
  --snp_files SNP_FILES [SNP_FILES ...]
                        label for SNPs for multiple diseases
  --frac_pretrain FRAC_PRETRAIN
                        Fractions of pre-training samples
  --frac_trains FRAC_TRAINS [FRAC_TRAINS ...]
                        Fractions of training samples
  --frac_test FRAC_TEST
                        Fraction of testing samples
  --frac_val FRAC_VAL   Fraction of validation_split
  --methods METHODS [METHODS ...]
                        Methods used in comparison
  --nrep NREP           Number for experiments
  --batch_size BATCH_SIZE
                        The batch size for training
  --epochs EPOCHS       The max epoch for training
  --dropout DROPOUT     The dropout rate for training
  --ilayer ILAYER       ith layer starting transfer learning
  --nodes NODES [NODES ...]
                        Number of nodes in customized layers

```


## Example


### Obtain flanking DNA sequence using chromosome coordinates
```
Rscript --vanilla snptoseq.R HGMD HGMD 500 hg19
```

### Pre-train model using large-scale generic functional noncoding variants
```
python main.py \
    --pretrain \
    --jobname hgmd \
    --seq_files  HGMD.fasta \
    --snp_files  HGMD.bed \
    --frac_pretrain 1  \
    --frac_val 0.2  \
    --batch_size 128  \
    --epochs 50  
```



### Re-train  model using context-specific functional noncoding variants and compare four deep learning models:
(i) Base-model: all layers of the base network is pretrained by integrated generic functional NCVs; (ii) Self-model: all layers of the base network is trained by context-specific MPRA variants; (iii) Transfer learning model (TLVar): the convolutional layers are inherited from the base network with all layers’ parameters frozen and the dense layers are retrained by context-specific MPRA variants in the target network; (iv) Transfer learning model without retraining dense layers (TLVar0): the convolutional layers are inherited from the base network with all layers’ parameters frozen but dense layers are randomly initiated. 


```
python main.py \
    --comparison\
    --jobname CAGI_train \
    --pretrain_model hgmd1.0.seq.h5  \
    --data_file CAGI_train.h5 \
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
```


