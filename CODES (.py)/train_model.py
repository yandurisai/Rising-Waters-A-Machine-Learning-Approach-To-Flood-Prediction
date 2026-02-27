"""
Flood Prediction Model Training
- EDA
- Preprocessing
- Multiple Classification Algorithms
- Model Evaluation & Saving
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

# For Bootstrap
from sklearn.base import clone

# ─────────────────────────────────────────────
# 1. LOAD / GENERATE DATA
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'flood_data.csv')
MODELS_PATH = os.path.join(os.path.dirname(__file__), 'models')
STATIC_PATH = os.path.join(os.path.dirname(__file__), 'static')

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(os.path.join(STATIC_PATH, 'plots'), exist_ok=True)

if not os.path.exists(DATA_PATH):
    print("Generating dataset...")
    from data.generate_dataset import generate_flood_dataset
    df = generate_flood_dataset(5000)
    df.to_csv(DATA_PATH, index=False)
else:
    df = pd.read_csv(DATA_PATH)

print(f"\n{'='*60}")
print("FLOOD PREDICTION - ML PIPELINE")
print(f"{'='*60}")
print(f"Dataset shape: {df.shape}")
print(f"Flood events: {df['flood'].sum()} ({df['flood'].mean():.1%})")

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n[1/5] Exploratory Data Analysis...")

# Distribution of target
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Flood Event Distribution', fontsize=14, fontweight='bold')

counts = df['flood'].value_counts()
axes[0].bar(['No Flood', 'Flood'], counts.values, color=['#2196F3', '#F44336'], alpha=0.85)
axes[0].set_title('Class Distribution')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

axes[1].pie(counts.values, labels=['No Flood', 'Flood'],
            colors=['#2196F3', '#F44336'], autopct='%1.1f%%', startangle=90)
axes[1].set_title('Class Proportion')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_PATH, 'plots', 'class_distribution.png'), dpi=100, bbox_inches='tight')
plt.close()

# Feature correlations
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
fig, ax = plt.subplots(figsize=(14, 10))
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, annot_kws={'size': 8})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_PATH, 'plots', 'correlation_matrix.png'), dpi=100, bbox_inches='tight')
plt.close()

# Feature distributions by flood class
features_to_plot = ['rainfall_mm', 'river_level_m', 'soil_moisture_pct', 'humidity_pct',
                    'temperature_c', 'elevation_m']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Feature Distributions by Flood Class', fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, feature in enumerate(features_to_plot):
    flood_yes = df[df['flood'] == 1][feature]
    flood_no = df[df['flood'] == 0][feature]
    axes[i].hist(flood_no, bins=30, alpha=0.6, color='#2196F3', label='No Flood', density=True)
    axes[i].hist(flood_yes, bins=30, alpha=0.6, color='#F44336', label='Flood', density=True)
    axes[i].set_title(feature.replace('_', ' ').title())
    axes[i].legend()
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Density')

plt.tight_layout()
plt.savefig(os.path.join(STATIC_PATH, 'plots', 'feature_distributions.png'), dpi=100, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 3. DATA PREPROCESSING
# ─────────────────────────────────────────────
print("[2/5] Data Preprocessing...")

# Drop date column
df_model = df.drop(columns=['date'])

# Feature engineering
df_model['rainfall_river_interaction'] = df_model['rainfall_mm'] * df_model['river_level_m']
df_model['low_elevation_near_river'] = ((df_model['elevation_m'] < 20) &
                                         (df_model['distance_to_river_km'] < 2)).astype(int)
df_model['high_risk_conditions'] = ((df_model['rainfall_mm'] > 30) &
                                     (df_model['soil_moisture_pct'] > 60)).astype(int)

# Features and target
FEATURE_COLS = [c for c in df_model.columns if c != 'flood']
X = df_model[FEATURE_COLS]
y = df_model['flood']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Test set: {X_test.shape[0]} samples")

# Save scaler & feature names
joblib.dump(scaler, os.path.join(MODELS_PATH, 'scaler.pkl'))
with open(os.path.join(MODELS_PATH, 'feature_names.json'), 'w') as f:
    json.dump(FEATURE_COLS, f)

# ─────────────────────────────────────────────
# 4. TRAIN MULTIPLE MODELS
# ─────────────────────────────────────────────
print("[3/5] Training Models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')

    results[name] = {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_prob': y_prob
    }
    print(f"    Accuracy={results[name]['accuracy']:.3f}, F1={results[name]['f1']:.3f}, AUC={results[name]['roc_auc']:.3f}")

# ─────────────────────────────────────────────
# 5. BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────
print("[4/5] Bootstrap Confidence Intervals (Random Forest)...")

best_model_name = max(results, key=lambda k: results[k]['roc_auc'])
best_model = results[best_model_name]['model']

n_bootstrap = 100
bootstrap_aucs = []
for i in range(n_bootstrap):
    X_bs, y_bs = resample(X_test_scaled, y_test, random_state=i)
    probs = best_model.predict_proba(X_bs)[:, 1]
    try:
        bootstrap_aucs.append(roc_auc_score(y_bs, probs))
    except:
        pass

ci_lower = np.percentile(bootstrap_aucs, 2.5)
ci_upper = np.percentile(bootstrap_aucs, 97.5)
print(f"  Best model: {best_model_name}")
print(f"  AUC: {np.mean(bootstrap_aucs):.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})")

# Bootstrap plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(bootstrap_aucs, bins=20, color='#4CAF50', alpha=0.8, edgecolor='white')
ax.axvline(ci_lower, color='red', linestyle='--', linewidth=2, label=f'95% CI Lower: {ci_lower:.3f}')
ax.axvline(ci_upper, color='blue', linestyle='--', linewidth=2, label=f'95% CI Upper: {ci_upper:.3f}')
ax.axvline(np.mean(bootstrap_aucs), color='black', linewidth=2, label=f'Mean AUC: {np.mean(bootstrap_aucs):.3f}')
ax.set_title(f'Bootstrap Distribution of AUC - {best_model_name}', fontsize=12, fontweight='bold')
ax.set_xlabel('AUC Score')
ax.set_ylabel('Frequency')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(STATIC_PATH, 'plots', 'bootstrap_auc.png'), dpi=100, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 6. VISUALIZATION & EVALUATION
# ─────────────────────────────────────────────
print("[5/5] Generating Evaluation Plots...")

# Model comparison
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
model_names = list(results.keys())
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(model_names))
width = 0.15
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']

for i, metric in enumerate(metrics):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(x + i * width, vals, width, label=metric.upper().replace('_', ' '),
                  color=colors[i], alpha=0.85)

ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.legend(loc='lower right')
ax.set_ylim(0, 1.1)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_PATH, 'plots', 'model_comparison.png'), dpi=100, bbox_inches='tight')
plt.close()

# ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))
colors_roc = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
for (name, res), color in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name} (AUC={res['roc_auc']:.3f})")

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_PATH, 'plots', 'roc_curves.png'), dpi=100, bbox_inches='tight')
plt.close()

# Confusion Matrix for best model
fig, ax = plt.subplots(figsize=(7, 6))
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Flood', 'Flood'],
            yticklabels=['No Flood', 'Flood'])
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_PATH, 'plots', 'confusion_matrix.png'), dpi=100, bbox_inches='tight')
plt.close()

# Feature Importance (Random Forest)
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 8))
colors_fi = ['#F44336' if imp > importances.median() else '#2196F3' for imp in importances]
importances.plot(kind='barh', ax=ax, color=colors_fi, alpha=0.85)
ax.set_title('Feature Importances - Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_PATH, 'plots', 'feature_importance.png'), dpi=100, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 7. SAVE BEST MODEL & METRICS
# ─────────────────────────────────────────────
joblib.dump(best_model, os.path.join(MODELS_PATH, 'best_model.pkl'))

metrics_summary = {
    'best_model': best_model_name,
    'bootstrap_auc_mean': float(np.mean(bootstrap_aucs)),
    'bootstrap_ci_lower': float(ci_lower),
    'bootstrap_ci_upper': float(ci_upper),
    'models': {
        name: {k: float(v) for k, v in res.items()
               if k not in ['model', 'y_pred', 'y_prob']}
        for name, res in results.items()
    }
}
with open(os.path.join(MODELS_PATH, 'metrics.json'), 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print(f"\n{'='*60}")
print(f"✓ Best Model: {best_model_name}")
print(f"  Accuracy  : {results[best_model_name]['accuracy']:.3f}")
print(f"  F1 Score  : {results[best_model_name]['f1']:.3f}")
print(f"  ROC-AUC   : {results[best_model_name]['roc_auc']:.3f}")
print(f"  Bootstrap AUC: {np.mean(bootstrap_aucs):.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
print(f"\n✓ Models saved to: {MODELS_PATH}")
print(f"✓ Plots saved to:  {STATIC_PATH}/plots")
print(f"{'='*60}")
