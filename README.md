
# 🤖 Maintenance Prédictive & Diagnostic de Défauts par IA pour Moteur CC (MCC)

Ce projet met en œuvre une approche basée sur le **Machine Learning** et l'**Ingénierie Avancée de Descripteurs (Feature Engineering)** pour la détection et l'isolation de défauts (FDI - *Fault Detection and Isolation*) sur un Moteur à Courant Continu (MCC).

Le modèle utilisé, un **ExtraTreesClassifier** optimisé par recherche d'hyperparamètres (*RandomizedSearchCV*), permet de classifier **9 types d'états de fonctionnement** (sain et défaillances) avec une précision globale (**Accuracy**) de **95%**.

---

## 📊 Performances du Modèle

* **Accuracy globale :** 95.00 %
* **F1-Score Macro :** 94.93 %
* **Matthews Correlation Coefficient (MCC) :** 0.9440
* **Validation Croisée Out-Of-Fold (OOF) :** 95.42 % *(Garantie d'absence d'overfitting)*

### 🏆 Détection par classe
* **Précision parfaite (F1 = 100%) :** Capteur Courant, Capteur Vitesse, Défaut Inductance ($L$), Charge (*Load*), Résistance Statorique ($R$), Roulement.
* **Détection haute précision (> 80%) :** Défaut d'Inertie ($J$), Résistance Rotorique ($Rei$), État Nominal.

---

## ⚙️ Classes de Défauts Traitées

1. **`Nominal`** : Fonctionnement sain du moteur.
2. **`Capteur_Courant`** : Dysfonctionnement / Bruit du capteur de courant.
3. **`Capteur_Vitesse`** : Dysfonctionnement / Bruit du capteur de vitesse.
4. **`Defaut_R`** : Variation anormale de la résistance statorique.
5. **`Defaut_L`** : Variation de l'inductance statorique.
6. **`Defaut_Rei`** : Dérive de la résistance rotorique / circuit d'excitation.
7. **`Defaut_J`** : Variation du moment d'inertie de l'arbre.
8. **`Defaut_Load`** : Perturbation de la charge mécanique.
9. **`Defaut_Roulement`** : Signature vibratoire / mécanique de dégradation du roulement.

---

## 🛠️ Descripteurs Extraits (Feature Engineering)

Pour différencier les micro-défauts, le script génère **47 descripteurs temporels et statistiques** avancés à partir de deux voies de mesure (`r1` et `r2`) :

* **Descripteurs différentiels :** Écart direct (`Diff`), absolu (`AbsDiff`) et relatif (`RelDiff`) des métriques RMS, Peak, Mean, Std et Kurtosis.
* **Facteurs de forme & d'impulsion :** *Crest Factor*, *Shape Factor*, *Impulse Factor*, *Margin Factor*.
* **Énergies & Variances :** Analyse des niveaux d'énergie dynamique (`Energy_Diff`, `Energy_Ratio`).
* **Signatures d'oscillation :** Couplage de la variance et de l'aplatissement (Kurtosis) pour capturer les frottements mécaniques et les transitoires.

Les variables ayant le plus fort pouvoir discriminant sont :
1. `Energy_Diff` (12.6 %)
2. `Diff_RMS` (7.2 %)
3. `AbsDiff_RMS` (6.8 %)
4. `RelDiff_Kurtosis` (6.3 %)

---

## 💻 Structure du Script & Technologies

* **Langage :** Python 3.x
* **Bibliothèques principales :** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Algorithme :** `ExtraTreesClassifier` (Ensemble Learning)
* **Stratégie d'évaluation :** Validation croisée stratifiée à 5 plis (*StratifiedKFold*) + Validation finale *Out-Of-Fold*.

---

## 🚀 Utilisation

1. Placez votre dataset (`dataset_fdi6.csv`) sur le bureau ou modifiez le chemin `desktop_path` dans le script.
2. Exécutez le script principal :
   ```bash
   python main_fdi_mcc.py
