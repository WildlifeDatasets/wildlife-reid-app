import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import faiss
import timm  # MegaDescriptoru
import os


class EmbeddingEngine:
    def __init__(self, device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Loading MegaDescriptor on {self.device}...")

        # NAČTENÍ SKUTEČNÉHO MEGADESCRIPToru z Hugging Face
        # num_classes=0 zajistí, že model vrátí čisté embeddingy (vlastnosti), ne klasifikaci
        self.model = timm.create_model(
            "hf-hub:BVRA/MegaDescriptor-T-224", pretrained=True, num_classes=0
        )

        self.model.to(self.device)
        self.model.eval()

        # MegaDescriptor-T používá standardní ImageNet transformace
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                ),
            ]
        )

        self.index = None
        self.embeddings = None

    def generate_embeddings(self, image_paths, bboxes=None, progress_callback=None):
        embeddings = []
        total = len(image_paths)

        with torch.no_grad():
            for i, path in enumerate(image_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    
                    # Handle cropping if bbox is provided
                    bbox = bboxes[i] if bboxes is not None else None
                    if bbox:
                        try:
                            w, h = img.size
                            x1, y1, x2, y2 = bbox
                            
                            # Add 10% padding
                            bw = x2 - x1
                            bh = y2 - y1
                            pad_w = bw * 0.1
                            pad_h = bh * 0.1
                            
                            x1 = max(0, x1 - pad_w)
                            y1 = max(0, y1 - pad_h)
                            x2 = min(w, x2 + pad_w)
                            y2 = min(h, y2 + pad_h)
                            
                            img = img.crop((x1, y1, x2, y2))
                        except Exception as e:
                            print(f"Cropping failed for {path}, using full image: {e}")

                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    # Model vrátí vektor vlastností
                    feat = self.model(img_tensor).cpu().numpy().flatten()
                    embeddings.append(feat)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    # Pokud obrázek nejde načíst, vložíme prázdný vektor
                    if len(embeddings) > 0:
                        embeddings.append(np.zeros_like(embeddings[0]))
                    else:
                        embeddings.append(np.zeros(768))

                if progress_callback:
                    progress_callback(int((i + 1) / total * 100))

        self.embeddings = np.array(embeddings).astype("float32")
        # Normalizace pro kosinovou podobnost (klíčové pro Re-ID)
        faiss.normalize_L2(self.embeddings)

        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(
            d
        )  # IndexFlatIP + normalizace = Kosinová podobnost
        self.index.add(self.embeddings)
        return self.embeddings

    def find_nearest_neighbors(self, k=5):
        if self.index is None:
            return None, None
        # Najde k+1 sousedů (první je vždy ten samý obrázek)
        D, I = self.index.search(self.embeddings, k + 1)
        return D, I