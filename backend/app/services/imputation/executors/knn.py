from typing import Dict
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

def run(df: pd.DataFrame, params: Dict):
    print("\n[DEBUG-KNN] --- DÉBUT DE L'IMPUTATION K-NN ---")
    
    # ==================== NOUVEAU BLOC DE CORRECTION ====================
    # On force manuellement les colonnes qui contiennent du texte à être de type 'object'.
    # Cela évite que Pandas se trompe de type de données à la lecture du CSV.
    for col in df.columns:
        # Si une valeur dans la colonne (en ignorant les NaN) est une chaîne de caractères,
        # alors toute la colonne doit être traitée comme du texte (object).
        if pd.api.types.is_string_dtype(df[col]):
            continue # Déjà du texte, on ne fait rien
        
        # On vérifie si, après avoir enlevé les NaN, il reste des chaînes de caractères
        is_object = df[col].dropna().apply(lambda x: isinstance(x, str)).any()
        if is_object:
            print(f"[DEBUG-KNN] Forçage de la colonne '{col}' en type 'object'.")
            # Remplacer les chaînes vides ou les espaces par de vrais NaN
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True) 
            df[col] = df[col].astype('object')
    # ====================================================================

    n_neighbors = params.get("n_neighbors", 5)
    weights = params.get("weights", "uniform")

    # 1. Séparer les types de colonnes (devrait fonctionner correctement maintenant)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    print(f"[DEBUG-KNN] Colonnes numériques détectées: {numeric_cols}")
    print(f"[DEBUG-KNN] Colonnes catégorielles détectées: {categorical_cols}")

    if not categorical_cols:
        print("[DEBUG-KNN] AVERTISSEMENT: Aucune colonne catégorielle détectée. L'imputation ne se fera que sur les nombres.")
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        df_filled = df.copy()
        if numeric_cols:
            df_filled[numeric_cols] = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)
        return df_filled
        
    df_filled = df.copy()

    # 2. Encoder les colonnes catégorielles
    print("[DEBUG-KNN] Encodage des colonnes catégorielles avec pd.get_dummies...")
    df_encoded = pd.get_dummies(df, columns=categorical_cols, dummy_na=True, dtype=float)
    print(f"[DEBUG-KNN] Aperçu des colonnes encodées: {df_encoded.columns.tolist()}")

    # 3. Exécuter l'imputation
    print("[DEBUG-KNN] Exécution de KNNImputer...")
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    df_imputed_encoded_np = imputer.fit_transform(df_encoded)
    df_imputed_encoded = pd.DataFrame(df_imputed_encoded_np, columns=df_encoded.columns)
    print("[DEBUG-KNN] Imputation terminée.")

    # 4. Décoder les résultats
    print("[DEBUG-KNN] --- DÉBUT DU DÉCODAGE ---")
    for col in categorical_cols:
        print(f"\n[DEBUG-KNN] Décodage de la colonne: '{col}'")
        dummy_columns = [c for c in df_imputed_encoded.columns if c.startswith(f"{col}_")]
        
        if not dummy_columns:
            continue

        imputed_series = df_imputed_encoded[dummy_columns].idxmax(axis=1)

        def clean_label(label):
            prefix = f"{col}_"
            if str(label).endswith("_nan"):
                return np.nan
            return str(label)[len(prefix):]

        df_filled[col] = imputed_series.apply(clean_label).astype(df[col].dtype)

    # 5. Mettre à jour les colonnes numériques
    print("\n[DEBUG-KNN] Mise à jour des colonnes numériques.")
    df_filled[numeric_cols] = df_imputed_encoded[numeric_cols]

    print("[DEBUG-KNN] --- FIN DE L'IMPUTATION K-NN ---")
    return df_filled
