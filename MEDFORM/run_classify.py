import pandas as pd
import torch
import os
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

from core.utils.learning import set_seed

import torch.nn as nn
import torch.optim as optim

# ACRIN_BREAST_TASKS 설정
ACRIN_BREAST_TASKS = ['T_stage_binary']
BREAST_TASKS = {'NSCLC_Radiogenomics_fixed': ACRIN_BREAST_TASKS}


class MLPClassifier(nn.Module):  # MLP 분류기 정의
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def calculate_metrics(y_true, y_pred, pred_scores):
    auc = roc_auc_score(y_true, pred_scores, multi_class="ovr", average="macro") if len(np.unique(y_true)) > 2 else roc_auc_score(y_true, pred_scores[:, 1])
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return auc, acc, bacc, f1


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

    # intersection = list(set(labels['slide_id'].values.tolist()) & set(slide_ids))
    intersection = list(set(labels['Case ID'].values.tolist()) & set(slide_ids))
    # labels = labels[labels['slide_id'].isin(intersection)]
    labels = labels[labels['Case ID'].isin(intersection)]
    num_classes = len(labels[study].unique())

    # X = np.array([embeddings[n] for n in labels['slide_id']])
    X = np.array([embeddings[n] for n in labels['Case ID']])
    y = labels[study].values

    return X, y


def save_results_to_txt(file_path, fold, metrics, cm):
    with open(file_path, 'w') as f:
        f.write(f"Results for Fold {fold}:\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy: {metrics['bacc']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1']:.4f}\n")
        f.write(f"AUROC: {metrics['auc']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm, separator=', '))
        f.write("\n")


def eval_single_task(DATASET_NAME, TASKS, PATH, verbose=True):
    if DATASET_NAME == "NSCLC_Radiogenomics_fixed":
        EMBEDS_PATH = "/home/alleun1110/Digital_bio/TANGLE_renew/results/NSCLC_Radiogenomics_fixed_checkpoints_and_embeddings/SimCLR_CT_2_pretrained_tangle_NSCLC_Radiogenomics_fixed_T_stage_clinical_RNA_lr0.0001_epochs100_bs64_tokensize2048_temperature0.01_uni/NSCLC_Radiogenomics_fixed_results_dict_pretrained.pkl"
        LABEL_PATH = '/home/alleun1110/Digital_bio/TANGLE_renew/dataset_csv/NSCLC_labels_few_shot_T_stage_fixed.csv'
    else:
        raise NotImplementedError("Dataset not implemented")

    LABELS = pd.read_csv(LABEL_PATH)
    LABELS['Case ID'] = LABELS['Case ID'].astype(str)
    BASE_OUT = '/'.join(EMBEDS_PATH.split('/')[:-1])

    for task in TASKS:
        print(f"Task {task}...")
        X, y = load_and_split(LABELS, EMBEDS_PATH, task)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold = 1

        for train_index, test_index in skf.split(X, y):
            print(f"Train : {train_index}")
            print(f"Test : {test_index}")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
            )
            input_dim = X_train.shape[1]
            hidden_dim = 128
            num_classes = len(np.unique(y_train))

            model = MLPClassifier(input_dim, hidden_dim, num_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            train_features = torch.from_numpy(X_train).float()
            train_labels = torch.from_numpy(y_train).long()
            test_features = torch.from_numpy(X_test).float()
            test_labels = torch.from_numpy(y_test).long()

            num_epochs = 50
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(train_features)
                loss = criterion(outputs, train_labels)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                test_outputs = model(test_features)
                _, pred_labels = torch.max(test_outputs, 1)
                pred_scores = torch.softmax(test_outputs, dim=1).numpy()

            auc, acc, bacc, f1 = calculate_metrics(test_labels.numpy(), pred_labels.numpy(), pred_scores)
            cm = confusion_matrix(test_labels.numpy(), pred_labels.numpy())

            results_path = os.path.join(BASE_OUT, f"test_CT_pretrained_results_fold_{fold}.txt")
            save_results_to_txt(results_path, fold, {"accuracy": acc, "bacc": bacc, "f1": f1, "auc": auc}, cm)
            print(f"Results saved for Fold {fold} in {results_path}")

            fold += 1


if __name__ == "__main__":
    tasks = BREAST_TASKS
    print("* Evaluating on breast...")
    MODELS = {
        # 'tangle_ACRIN_Breast': "/home/alleun1110/Digital_bio/TANGLE_renew/results/ACRIN_Breast_checkpoints_and_embeddings/SimCLR-PET-CT_2_pretrained_tangle_ACRIN_Breast_T_stage_clinical_RNA_lr0.0001_epochs100_bs64_tokensize2048_temperature0.01_uni"
        'tangle_NSCLC_Radiogenomics' : "/home/alleun1110/Digital_bio/TANGLE_renew/results/NSCLC_Radiogenomics_fixed_checkpoints_and_embeddings/SimCLR_CT_2_pretrained_tangle_NSCLC_Radiogenomics_fixed_T_stage_clinical_RNA_lr0.0001_epochs100_bs64_tokensize2048_temperature0.01_uni"
    }

    for exp_name, p in MODELS.items():
        for n, t in tasks.items():
            print(f'\n* Dataset: {n}')
            eval_single_task(n, t, p, verbose=True)
