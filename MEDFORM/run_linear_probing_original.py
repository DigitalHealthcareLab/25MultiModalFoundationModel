import pandas as pd
import torch
import os
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score, confusion_matrix, accuracy_score, f1_score
from core.utils.learning import set_seed

ACRIN_BREAST_TASKS = ['label']
BREAST_TASKS = {'ACRIN': ACRIN_BREAST_TASKS}

def print_confusion_matrix(cm):
    print("Confusion Matrix:")
    print("-----------------")
    for row in cm:
        print(" ".join(f"{x:4d}" for x in row))
    print()

def calculate_metrics(y_true, y_pred, pred_scores):
    if len(np.unique(y_true)) > 2:
        auc = roc_auc_score(y_true, pred_scores, multi_class="ovr", average="macro")
        f1 = f1_score(y_true, y_pred, average='macro')
    else:
        auc = roc_auc_score(y_true, pred_scores[:, 1])
        f1 = f1_score(y_true, y_pred)
    
    bacc = balanced_accuracy_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    return auc, bacc, acc, f1, cm

def load_and_split(labels, embedding_path, study, k=1, normalize=False):
    file = open(embedding_path, 'rb')
    obj = pickle.load(file)
    embeddings = obj['embeds']
    if normalize:
        pipe = Pipeline([('scaler', StandardScaler())])
        embeddings = pipe.fit_transform(embeddings)
    slide_ids = obj['slide_ids']
    slide_ids = [str(x) for x in slide_ids]
    embeddings = {n: e for e, n in zip(embeddings, slide_ids)}
    
    intersection = list(set(labels['slide_id'].values.tolist()) & set(slide_ids))
    labels = labels[labels['slide_id'].isin(intersection)]
    num_classes = len(labels[study].unique())
    
    train_slide_ids = []
    for cls in range(num_classes):
        train_slide_ids += labels[labels[study] == cls].sample(k)['slide_id'].values.tolist()
    test_slide_ids = labels[~labels['slide_id'].isin(train_slide_ids)]['slide_id'].values.tolist()
    
    train_embeddings = np.array([embeddings[n] for n in train_slide_ids])
    test_embeddings = np.array([embeddings[n] for n in test_slide_ids])
    train_labels = np.array([labels[labels['slide_id']==slide_id][study].values for slide_id in train_slide_ids])
    test_labels = np.array([labels[labels['slide_id']==slide_id][study].values for slide_id in test_slide_ids])
    
    train_embeddings = torch.from_numpy(train_embeddings)
    test_embeddings = torch.from_numpy(test_embeddings)
    train_labels = torch.from_numpy(train_labels).squeeze()
    test_labels = torch.from_numpy(test_labels).squeeze()
    
    if len(train_embeddings.shape) == 1:
        train_embeddings = torch.unsqueeze(train_embeddings, 0)
        train_labels = torch.unsqueeze(train_labels, 0)
    
    return train_embeddings, train_labels, test_embeddings, test_labels

def eval_single_task(DATASET_NAME, TASKS, PATH, verbose=True):
    ALL_K = [1, 5, 10]
    if DATASET_NAME == "ACRIN":
        EMBEDS_PATH = "/home/alleun1110/Digital_bio/TANGLE_renew/results/Colorectal_checkpoints_and_embeddings/Synthetic_TabNet_pretrained_tangle_Colorectal_T_stage_clinical_RNA_lr0.0001_epochs100_bs64_tokensize2048_temperature0.01_uni/Colorectal_results_dict_pretrained.pkl"
        LABEL_PATH = '/home/alleun1110/Digital_bio/TANGLE_renew/dataset_csv/Colorectal_cancer/CRC_bilobar_label.csv'
    else:
        raise NotImplementedError("Dataset not implemented")
    
    BASE_OUT = '/'.join(EMBEDS_PATH.split('/')[:-1])
    
    for k in ALL_K:
        for task in TASKS:
            if verbose:
                print(f"Task {task} and k = {k}...")
            
            NUM_FOLDS = 5
            metrics_store_all = {}
            RESULTS_FOLDER = f"k={k}_probing_{task.replace('/', '')}"
            metrics_store = {"auc": [], "bacc": [], "acc": [], "f1": []}
            
            for fold in range(NUM_FOLDS):
                set_seed(SEED=fold)
                if verbose:
                    print(f" Going for fold {fold}...")
                
                LABELS = pd.read_csv(LABEL_PATH)
                LABELS['slide_id'] = LABELS['slide_id'].astype(str)
                LABELS = LABELS[LABELS[task] != -1]
                LABELS = LABELS[['slide_id', task]]
                
                train_features, train_labels, test_features, test_labels = load_and_split(LABELS, EMBEDS_PATH, task, k)
                
                if verbose:
                    print(f" Fitting logistic regression on {len(train_features)} slides")
                    print(f" Evaluating on {len(test_features)} slides")
                
                NUM_C = 2
                COST = (train_features.shape[1] * NUM_C) / 100
                clf = LogisticRegression(C=COST, max_iter=10000, verbose=0, random_state=0)
                clf.fit(X=train_features, y=train_labels)
                pred_labels = clf.predict(X=test_features)
                pred_scores = clf.predict_proba(X=test_features)
                
                if verbose:
                    print(" Updating metrics store...")
                
                auc, bacc, acc, f1, cm = calculate_metrics(test_labels.numpy(), pred_labels, pred_scores)
                metrics_store["auc"].append(auc)
                metrics_store["bacc"].append(bacc)
                metrics_store["acc"].append(acc)
                metrics_store["f1"].append(f1)
                
                if verbose:
                    print(f" Done for fold {fold}")
                    print(f" AUC: {round(auc, 3)}, BACC: {round(bacc, 3)}, ACC: {round(acc, 3)}, F1: {round(f1, 3)}")
                    print_confusion_matrix(cm)
                    print()
            
            metrics_store_all['tangle'] = metrics_store
            
            print(f'k={k}, task={task}')
            print(f'AUC: {round(np.array(metrics_store["auc"]).mean(), 3)} +/- {round(np.array(metrics_store["auc"]).std(), 3)}')
            print(f'BACC: {round(np.array(metrics_store["bacc"]).mean(), 3)} +/- {round(np.array(metrics_store["bacc"]).std(), 3)}')
            print(f'ACC: {round(np.array(metrics_store["acc"]).mean(), 3)} +/- {round(np.array(metrics_store["acc"]).std(), 3)}')
            print(f'F1: {round(np.array(metrics_store["f1"]).mean(), 3)} +/- {round(np.array(metrics_store["f1"]).std(), 3)}')
            
            os.makedirs(f'{BASE_OUT}/{DATASET_NAME}', exist_ok=True)
            with open(f'{BASE_OUT}/{DATASET_NAME}/{RESULTS_FOLDER}.pickle', 'wb') as handle:
                pickle.dump(metrics_store_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    tasks = BREAST_TASKS
    print("* Evaluating on breast...")
    print("* All datasets to evaluate on = {}".format(list(tasks.keys())))
    
    MODELS = {
        'tangle_ACRIN': "/home/alleun1110/Digital_bio/TANGLE_renew/results/Colorectal_checkpoints_and_embeddings/Synthetic_TabNet_pretrained_tangle_Colorectal_T_stage_clinical_RNA_lr0.0001_epochs100_bs64_tokensize2048_temperature0.01_uni"
    }
    
    for exp_name, p in MODELS.items():
        for n, t in tasks.items():
            print('\n* Dataset:', n)
            eval_single_task(n, t, p, verbose=True)
