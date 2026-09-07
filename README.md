# Détection de maladies foliaires par YOLO

Projet CV — fine-tuning d'un modèle YOLO pour détecter et classer des maladies foliaires sur des
images de plantes prises en conditions de terrain.

## Structure

```
exam/
├── app.py                  # Démo interactive Streamlit
├── data/                   # Dataset FieldPlant (téléchargé depuis Roboflow, non versionné)
├── notebooks/
│   └── finetuning_yolo.ipynb  # Notebook principal : EDA, entraînement, évaluation
├── reports/
│   ├── rapport.md          # Rapport d'expérimentation (source)
│   ├── rapport.docx        # Rapport au format Word (généré via pandoc)
│   └── figures/            # Figures exportées pour le rapport (dont figures/Demo/ pour l'app)
├── runs/                   # Sorties d'entraînement Ultralytics (non versionné)
├── src/                    # Scripts utilitaires (split, train, benchmark, qualitatif)
├── requirements.txt
└── .venv/                  # Environnement virtuel Python (non versionné)
```

## Dataset

[FieldPlant](https://universe.roboflow.com/plant-disease-detection/fieldplant) — images en conditions
réelles de champ, annotées en bounding boxes par des experts en pathologie végétale.
Téléchargé au format YOLOv8/v11 depuis Roboflow Universe et placé dans `data/`.

## Environnement

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

GPU utilisé : NVIDIA RTX 4060 (CUDA 12.4).

## Workflow

1. **EDA** : distribution des classes, exemples d'images annotées.
2. **Entraînement** : fine-tuning YOLO (comparaison nano vs small) sur FieldPlant.
3. **Évaluation** :
   - mAP@0.5, mAP@0.5:0.95
   - Précision / Rappel / F1-score par classe
   - Matrice de confusion
   - Courbes Précision-Rappel par classe
   - Temps d'inférence (ms/image) et taille du modèle
   - Exemples qualitatifs (bonnes détections, faux positifs, faux négatifs)
4. **Rapport** : synthèse des résultats dans `reports/`.

## Démo Streamlit

Une fois les modèles entraînés (`runs/detect/train_nano` et `train_small` présents) :

```bash
source .venv/Scripts/activate
streamlit run app.py
```

Permet d'uploader une image (ou de tirer un exemple aléatoire du test set), de choisir le modèle
(nano/small) et les seuils de confiance/IoU, et de visualiser les détections en direct.

