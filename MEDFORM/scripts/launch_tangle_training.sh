
# Train Tangle on the TCGA BRCA cohort. 
python train_tangle.py --study brca --method tangle
#python train_tangle.py --study ACRIN_Breast --method tangle
#python train_tangle.py --study NSCLC_Radiogenomics --method tangle

# Train Tangle-Rec on the TCGA BRCA cohort. 
python train_tangle.py --study brca --method tanglerec

# Train Intra on the TCGA BRCA cohort. 
python train_tangle.py --study brca --method intra

# To run few-shot evaluation
python extract_slide_embeddings_from_checkpoint.py --pretrained <PATH_TO_PRETRAINED_MODEL>
python run_linear_probing.py