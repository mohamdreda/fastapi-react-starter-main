from typing import Dict
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def run(df: pd.DataFrame, params: Dict):
    """
    Implémentation moderne de MissForest utilisant IterativeImputer de scikit-learn
    avec RandomForestRegressor comme estimateur.
    Gère les colonnes numériques et catégorielles.
    """
    print("\n[DEBUG-MISSFOREST] --- DÉBUT DE L'IMPUTATION MISSFOREST ---")
    
    # Paramètres avec valeurs par défaut
    n_estimators = params.get("n_estimators", 100)
    max_depth = params.get("max_depth", None)
    random_state = params.get("random_state", 42)
    max_iter = params.get("max_iter", 10)
    
    # Séparer les colonnes numériques et catégorielles
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"[DEBUG-MISSFOREST] Colonnes numériques: {numeric_cols}")
    print(f"[DEBUG-MISSFOREST] Colonnes catégorielles: {categorical_cols}")
    
    df_filled = df.copy()
    
    # Encoder les colonnes catégorielles
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Gérer les valeurs manquantes en les remplaçant temporairement
        mask = df_filled[col].notna()
        if mask.any():
            le.fit(df_filled[col][mask].astype(str))
            df_filled[col] = df_filled[col].astype(str)
            df_filled.loc[mask, col] = le.transform(df_filled[col][mask])
            df_filled[col] = pd.to_numeric(df_filled[col], errors='coerce')
            label_encoders[col] = le
            print(f"[DEBUG-MISSFOREST] Encodage de '{col}': {len(le.classes_)} classes")
    
    # Créer l'imputeur MissForest (IterativeImputer + RandomForest)
    rf_estimator = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    imputer = IterativeImputer(
        estimator=rf_estimator,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0
    )
    
    print("[DEBUG-MISSFOREST] Exécution de l'imputation...")
    
    # Effectuer l'imputation
    df_imputed_values = imputer.fit_transform(df_filled)
    df_imputed = pd.DataFrame(df_imputed_values, columns=df.columns, index=df.index)
    
    # Décoder les colonnes catégorielles
    for col in categorical_cols:
        if col in label_encoders:
            le = label_encoders[col]
            # Arrondir les valeurs imputées et les contraindre aux classes valides
            encoded_values = np.round(df_imputed[col]).astype(int)
            encoded_values = np.clip(encoded_values, 0, len(le.classes_) - 1)
            
            # Décoder vers les valeurs originales
            df_imputed[col] = le.inverse_transform(encoded_values)
            print(f"[DEBUG-MISSFOREST] Décodage de '{col}' terminé")
    
    # Restaurer les types de données originaux pour les colonnes numériques
    for col in numeric_cols:
        if col in df.columns:
            df_imputed[col] = df_imputed[col].astype(df[col].dtype)
    
    print("[DEBUG-MISSFOREST] --- FIN DE L'IMPUTATION MISSFOREST ---")
    return df_imputed
