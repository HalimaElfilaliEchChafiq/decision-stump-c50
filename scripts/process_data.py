import os
import pandas as pd
import sys

# Ajout du chemin pour importer les utils si besoin (optionnel ici)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

def process_file(filename):
    raw_path = os.path.join(parent_dir, 'data', 'raw', filename)
    processed_path = os.path.join(parent_dir, 'data', 'processed', filename)
    
    # 1. Vérifier si le fichier Raw existe
    if not os.path.exists(raw_path):
        print(f"⚠️  Attention : {filename} n'existe pas dans data/raw/ (Ignoré)")
        return

    print(f"🔄 Traitement de {filename}...")
    
    # 2. Chargement
    df = pd.read_csv(raw_path)
    
    # 3. Nettoyage (Simulation d'une tâche de Data Engineering)
    # On supprime les lignes complètement vides s'il y en a
    original_len = len(df)
    df.dropna(how='all', inplace=True)
    
    # On vérifie les types (Exemple: s'assurer que target est bien numérique)
    # (Ici c'est déjà le cas pour Iris/Wine, mais ça fait "pro" de le coder)
    if 'target' in df.columns:
        df['target'] = df['target'].astype(int)

    # 4. Sauvegarde dans Processed
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    
    print(f"✅ Sauvegardé dans data/processed/{filename} ({len(df)} lignes)")

if __name__ == "__main__":
    print("--- Début du Data Processing ---")
    
    # On traite les deux fichiers
    process_file('iris.csv')
    process_file('wine.csv')
    
    print("--- Terminé ---")