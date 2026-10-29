#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[8]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    RandomizedSearchCV, cross_val_predict
)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score,
    f1_score, matthews_corrcoef
)
from sklearn.feature_selection import VarianceThreshold


# ============================================================
# 1. CONFIGURATION
# ============================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
EPS = 1e-8

desktop_path = os.path.join(
    os.path.expanduser("~"), "Desktop", "dataset_fdi6.csv"
)

print(">>> [1/15] Script optimisé démarré...", flush=True)


# ============================================================
# 2. CHARGEMENT ET NETTOYAGE INITIAL
# ============================================================
if not os.path.exists(desktop_path):
    raise FileNotFoundError(f"Fichier introuvable sur le Bureau : {desktop_path}")

df = pd.read_csv(desktop_path)
print("=" * 65, flush=True)
print("DIMENSIONS INITIALES :", df.shape, flush=True)
print("=" * 65, flush=True)

df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

if "Fault_Class" not in df.columns:
    raise ValueError("La colonne 'Fault_Class' est absente du dataset.")

print("Distribution des classes :", flush=True)
print(df["Fault_Class"].value_counts(), flush=True)


# ============================================================
# 3. FEATURE ENGINEERING AVANCÉ (ANTI-CONFUSION NOMINAL/REI/L)
# ============================================================
print("\n>>> [3/15] Calcul du Feature Engineering avancé...", flush=True)

def safe_divide(a, b, eps=EPS):
    return np.divide(a, np.where(np.abs(b) < eps, eps, b))

# --- A. DESCRIPTEURS DIFFÉRENTIELS ET RELATIFS (Isolent les micro-défauts) ---
metrics = ['Mean', 'RMS', 'Peak', 'Std', 'Kurtosis']
for m in metrics:
    c1, c2 = f"{m}_r1", f"{m}_r2"
    if c1 in df.columns and c2 in df.columns:
        df[f"Diff_{m}"] = df[c1] - df[c2]
        df[f"AbsDiff_{m}"] = np.abs(df[c1] - df[c2])
        df[f"RelDiff_{m}"] = safe_divide(df[c1] - df[c2], np.abs(df[c1]) + np.abs(df[c2]))

# --- B. FACTEURS D'IMPULSION ET FORME ---
df["Impulse_Factor_r1"] = safe_divide(df["Peak_r1"], np.abs(df["Mean_r1"]))
df["Impulse_Factor_r2"] = safe_divide(df["Peak_r2"], np.abs(df["Mean_r2"]))

df["Shape_Factor_r1"] = safe_divide(df["RMS_r1"], np.abs(df["Mean_r1"]))
df["Shape_Factor_r2"] = safe_divide(df["RMS_r2"], np.abs(df["Mean_r2"]))

df["Margin_Factor_r1"] = safe_divide(df["Peak_r1"], df["RMS_r1"] ** 2)
df["Margin_Factor_r2"] = safe_divide(df["Peak_r2"], df["RMS_r2"] ** 2)

df["Crest_Factor_r1"] = safe_divide(df["Peak_r1"], df["RMS_r1"])
df["Crest_Factor_r2"] = safe_divide(df["Peak_r2"], df["RMS_r2"])

# --- C. VARIANCES ET ÉNERGIES DYNAMIQUES ---
df["Pure_Var_r1"] = np.maximum(0, df["RMS_r1"] ** 2 - df["Mean_r1"] ** 2)
df["Pure_Var_r2"] = np.maximum(0, df["RMS_r2"] ** 2 - df["Mean_r2"] ** 2)

df["Energy_r1"] = df["RMS_r1"] ** 2
df["Energy_r2"] = df["RMS_r2"] ** 2
df["Energy_Diff"] = df["Energy_r1"] - df["Energy_r2"]
df["Energy_Ratio"] = safe_divide(df["Energy_r1"], df["Energy_r2"])

# --- D. SIGNATURES D'OSCILLATION ET FROTTEMENT ---
df["Oscillation_L"] = df["Std_r1"] * df["Kurtosis_r1"]
df["Oscillation_J"] = df["Std_r2"] * df["Kurtosis_r2"]
df["Friction_Signature"] = df["Mean_r2"] * df["Std_r2"]

# --- E. RATIOS INTER-VOIES ---
df["Cross_Ratio_RMS"] = safe_divide(df["RMS_r1"], df["RMS_r2"])
df["Std_Ratio"] = safe_divide(df["Std_r1"], df["Std_r2"])
df["Peak_Ratio"] = safe_divide(df["Peak_r1"], df["Peak_r2"])
df["Mean_Ratio"] = safe_divide(df["Mean_r1"], df["Mean_r2"])
df["Kurtosis_Ratio"] = safe_divide(df["Kurtosis_r1"], df["Kurtosis_r2"])


# ============================================================
# 4. NETTOYAGE ET STRUCTURATION
# ============================================================
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

X = df.drop(columns=["Fault_Class"]).select_dtypes(include=[np.number])
y = df["Fault_Class"]

print(f"Nombre total de features générées : {X.shape[1]}", flush=True)


# ============================================================
# 5. TRAIN / TEST SPLIT STRATIFIÉ
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Train : {X_train.shape} | Test : {X_test.shape}", flush=True)


# ============================================================
# 6. FILTRAGE DES FEATURES INUTILES / CONSTANTES
# ============================================================
selector = VarianceThreshold(threshold=0.0)

X_train = pd.DataFrame(
    selector.fit_transform(X_train),
    columns=X_train.columns[selector.get_support()],
    index=X_train.index
)

X_test = pd.DataFrame(
    selector.transform(X_test),
    columns=X_train.columns,
    index=X_test.index
)

print(f"Features retenues après filtrage : {X_train.shape[1]}", flush=True)

warnings.filterwarnings("ignore")
# ============================================================
# 7. CONFIGURATION DU MODÈLE BASE
# ============================================================
base_model = ExtraTreesClassifier(
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=1
)


# ============================================================
# 8. OPTIMISATION FINE DES HYPERPARAMÈTRES
# ============================================================
param_distributions = {
    "n_estimators": [200, 300, 450],
    "max_depth": [None, 15, 25],
    "min_samples_split": [2, 3],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.6, 0.8],
    "criterion": ["gini", "entropy"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=12,
    scoring="f1_weighted",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=1,
    verbose=1,
    refit=True
)

print("\n" + "=" * 65, flush=True)
print(">>> [8/15] RECHERCHE DES MEILLEURS HYPERPARAMÈTRES...", flush=True)
print("=" * 65, flush=True)

search.fit(X_train, y_train)
model = search.best_estimator_

print("\nMeilleurs paramètres :", flush=True)
print(search.best_params_, flush=True)
print(f"Meilleur score CV : {search.best_score_ * 100:.2f}%\n", flush=True)


# ============================================================
# 9. ÉVALUATION SUR LE JEU DE TEST
# ============================================================
print(">>> [9/15] Évaluation sur le jeu de test...", flush=True)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred, average="weighted")
f1_macro = f1_score(y_test, y_pred, average="macro")
mcc = matthews_corrcoef(y_test, y_pred)

print("\n" + "=" * 65, flush=True)
print("PERFORMANCE FINALE DU MODÈLE OPTIMISÉ", flush=True)
print("=" * 65, flush=True)
print(f"Accuracy           : {accuracy * 100:.2f}%", flush=True)
print(f"Balanced Accuracy  : {balanced_acc * 100:.2f}%", flush=True)
print(f"F1-Score Weighted  : {f1_weighted * 100:.2f}%", flush=True)
print(f"F1-Score Macro     : {f1_macro * 100:.2f}%", flush=True)
print(f"MCC                : {mcc:.4f}", flush=True)
print("=" * 65, flush=True)


# ============================================================
# 10. RAPPORT DE CLASSIFICATION DÉTAILLÉ
# ============================================================
print("\nRAPPORT DE CLASSIFICATION PAR CLASSE", flush=True)
print("=" * 65, flush=True)
print(classification_report(y_test, y_pred, zero_division=0), flush=True)


# ============================================================
# 11. MATRICE DE CONFUSION (BRUTE)
# ============================================================
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Matrice de Confusion - ExtraTrees Optimisé", fontsize=13, fontweight="bold")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ============================================================
# 12. MATRICE DE CONFUSION (NORMALISÉE)
# ============================================================
cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_normalized = np.nan_to_num(cm_normalized)

plt.figure(figsize=(9, 7))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Matrice de Confusion Normalisée", fontsize=13, fontweight="bold")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ============================================================
# 13. IMPORTANCE DES FEATURES (TOP 20)
# ============================================================
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n" + "=" * 65, flush=True)
print("TOP 20 FEATURES LES PLUS DISCRIMINANTES", flush=True)
print("=" * 65, flush=True)
print(importance.head(20).to_string(index=False), flush=True)


# ============================================================
# 14. VISUALISATION DES FEATURES CLÉS
# ============================================================
top_features = importance.head(15).sort_values("Importance", ascending=True)
sns.barplot(data=top_features, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
plt.figure(figsize=(10, 7))
sns.barplot(data=top_features, x="Importance", y="Feature", palette="viridis")
plt.title("Top 15 Features Clés - ExtraTrees", fontsize=13, fontweight="bold")
plt.xlabel("Importance Relative")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# ============================================================
# 15. VALIDATION CROISÉE OUT-OF-FOLD
# ============================================================
print("\n>>> [15/15] Calcul de la validation croisée Out-Of-Fold finale...", flush=True)

cv_predictions = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=1)

cv_accuracy = accuracy_score(y_train, cv_predictions)
cv_f1_macro = f1_score(y_train, cv_predictions, average="macro")
cv_f1_weighted = f1_score(y_train, cv_predictions, average="weighted")

print("\n" + "=" * 65, flush=True)
print("VALIDATION CROISÉE FINALE (OOF)", flush=True)
print("=" * 65, flush=True)
print(f"CV Accuracy     : {cv_accuracy * 100:.2f}%", flush=True)
print(f"CV F1 Macro     : {cv_f1_macro * 100:.2f}%", flush=True)
print(f"CV F1 Weighted  : {cv_f1_weighted * 100:.2f}%", flush=True)
print("=" * 65, flush=True)


# In[ ]:





# In[ ]:




