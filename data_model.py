import pandas as pd
import numpy as np
import os
import faiss
import ast
from sklearn.model_selection import GroupShuffleSplit

class DataManager:
    def __init__(self):
        self.df = None
        self.image_root = ""
        
    def get_bboxes(self):
        """
        Parses 'detection_results' column and returns a list of bboxes.
        Returns None for images without valid detection.
        """
        if self.df is None or 'detection_results' not in self.df.columns:
            return [None] * len(self.df) if self.df is not None else []

        bboxes = []
        for _, row in self.df.iterrows():
            val = row['detection_results']
            bbox = None
            try:
                if pd.notna(val) and str(val).strip():
                    # Parse string representation of list of dicts
                    data = ast.literal_eval(val)
                    if isinstance(data, list) and len(data) > 0:
                        det = data[0]
                        if 'bbox' in det:
                            bbox = det['bbox'] # [x1, y1, x2, y2]
            except (ValueError, SyntaxError):
                pass
            bboxes.append(bbox)
        return bboxes

    def _normalize_id(self, val):
        if pd.isna(val):
            return "Unknown"
        # Convert to string and strip spaces
        s = str(val).strip()
        # Remove .0 if it's a float-like string (e.g., "1832.0" -> "1832")
        if s.endswith('.0'):
            s = s[:-2]
        return s

    def load_metadata(self, file_path, image_root):
        self.file_path = file_path
        self.image_root = image_root
        if file_path.endswith('.xlsx'):
            self.df = pd.read_excel(file_path)
        else:
            self.df = pd.read_csv(file_path)
            
        # 1. Prioritize 'unique_name' for ID extraction
        if 'unique_name' in self.df.columns:
            def extract_from_unique(val):
                try:
                    s = str(val).strip()
                    if '_' in s:
                        return self._normalize_id(s.split('_')[-1])
                    return self._normalize_id(s)
                except:
                    return "Unknown"
            self.df['Identity ID'] = self.df['unique_name'].apply(extract_from_unique)
        # 2. Check if 'Identity ID' already exists (and we didn't use unique_name)
        elif 'Identity ID' in self.df.columns:
            self.df['Identity ID'] = self.df['Identity ID'].apply(self._normalize_id)
        else:
            # 3. Extract from filename (fallback)
            def extract_id(filename):
                try:
                    name_no_ext = os.path.splitext(filename)[0]
                    return self._normalize_id(name_no_ext.split('_')[-1])
                except:
                    return "Unknown"
            self.df['Identity ID'] = self.df['filename'].apply(extract_id)
        
        # Full path to image
        self.df['full_path'] = self.df['filename'].apply(lambda x: os.path.join(self.image_root, x))
        self.df['exists'] = self.df['full_path'].apply(os.path.exists)
        return self.df

    def detect_errors(self, D, I, threshold=0.9):
        """
        D: similarities (cosine similarity 0-1)
        I: indices of nearest neighbors
        threshold: minimum similarity to flag as a potential error (different ID)
        """
        potential_errors = []
        
        for i in range(len(self.df)):
            query_id = self._normalize_id(self.df.iloc[i]['Identity ID'])
            neighbor_indices = I[i]
            similarities = D[i]
            
            # We only care about the SINGLE BEST match (excluding the image itself)
            for neighbor_idx, similarity in zip(neighbor_indices, similarities):
                if neighbor_idx == i:
                    continue
                
                # This is the absolute closest neighbor
                neighbor_id = self._normalize_id(self.df.iloc[neighbor_idx]['Identity ID'])
                
                # FLAG ONLY IF:
                # 1. The best match is a DIFFERENT identity
                # 2. AND the similarity is very high (above threshold)
                if query_id != neighbor_id and similarity > threshold:
                    potential_errors.append(i)
                
                # Important: we stop after checking the first non-self neighbor!
                # If the best match is the same ID, we don't care about others.
                break
                    
        self.df['potential_error'] = False
        self.df.loc[potential_errors, 'potential_error'] = True
        return potential_errors

    def update_id(self, index, new_id):
        self.df.at[index, 'Identity ID'] = self._normalize_id(new_id)

    def export_csv(self, output_path):
        self.df.to_csv(output_path, index=False)

    def split_data(self):
        if self.df is None:
            return None, None

        self.df["Split"] = "Train"  # Všechny nastavíme jako základ na Train

        # Seskupíme obrázky podle identity
        for uid, group in self.df.groupby("Identity ID"):
            if len(group) >= 2:
                # Pokud máme aspoň 2 fotky, jednu náhodně vybereme do Testu
                test_idx = np.random.choice(group.index)
                self.df.at[test_idx, "Split"] = "Test"
            else:
                # Pokud je tam jen jedna fotka, necháme ji v Trainu
                # (nemůžeme ji testovat, protože ji není s čím porovnat)
                pass

        train_idx = self.df[self.df["Split"] == "Train"].index
        test_idx = self.df[self.df["Split"] == "Test"].index
        return train_idx, test_idx

    def run_closest_set_validation(self, engine):
        if "Split" not in self.df.columns:
            return "Nejdříve spusťte Split Train/Test!"

        # 1. Rozdělení na Test (Query) a Train (Gallery)
        test_df = self.df[self.df["Split"] == "Test"]
        train_df = self.df[self.df["Split"] == "Train"]

        test_indices = test_df.index.tolist()
        train_indices = train_df.index.tolist()

        if not train_indices or not test_indices:
            return "Chybí data v Train nebo Test sadě."

        # 2. Vytvoření dočasného indexu jen z TRAIN embeddingů
        train_embeddings = engine.embeddings[train_indices]
        d = train_embeddings.shape[1]
        temp_index = faiss.IndexFlatIP(d)
        temp_index.add(train_embeddings)

        # 3. Vyhledávání TEST embeddingů v TRAIN indexu
        test_embeddings = engine.embeddings[test_indices]
        D, I = temp_index.search(test_embeddings, 1)

        correct = 0
        results = []
        for i in range(len(test_indices)):
            query_idx = test_indices[i]
            # Pozor: I[i][0] je index v rámci train_embeddings,
            # musíme ho namapovat na skutečný index v původním DF přes train_indices
            match_in_train_subset = I[i][0]
            match_idx = train_indices[match_in_train_subset]

            # Převedeme obojí na string a zbavíme se mezer pro jistotu porovnání
            q_id = str(self.df.at[query_idx, "Identity ID"]).strip()
            m_id = str(self.df.at[match_idx, "Identity ID"]).strip()

            # Odstranění ".0", pokud se ID načetlo jako float (časté u Excelu)
            if q_id.endswith(".0"):
                q_id = q_id[:-2]
            if m_id.endswith(".0"):
                m_id = m_id[:-2]

            similarity = float(D[i][0])
            is_correct = q_id == m_id

            if is_correct:
                correct += 1

            results.append(
                {
                    "query_idx": query_idx,
                    "match_idx": match_idx,
                    "score": similarity,
                    "correct": is_correct,
                }
            )

        accuracy = (correct / len(test_indices)) * 100 if len(test_indices) > 0 else 0
        return accuracy, results