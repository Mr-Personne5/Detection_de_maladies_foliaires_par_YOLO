import random
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "test" / "images"

MODELS = {
    "YOLOv8n (léger, recommandé)": RUNS_DIR / "train_nano" / "weights" / "best.pt",
    "YOLOv8s": RUNS_DIR / "train_small" / "weights" / "best.pt",
}

st.set_page_config(page_title="Détection de maladies foliaires", layout="wide")


@st.cache_resource
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def run_detection(model: YOLO, image: Image.Image, conf: float, iou: float):
    result = model.predict(source=image, conf=conf, iou=iou, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
    return annotated, result


st.title("🌿 Détection de maladies foliaires (YOLO)")
st.caption(
    "Démo du modèle YOLO fine-tuné sur le dataset FieldPlant (manioc, maïs, tomate — 27 classes)."
)

with st.sidebar:
    st.header("Paramètres")
    model_choice = st.selectbox("Modèle", list(MODELS.keys()))
    conf_threshold = st.slider("Seuil de confiance", 0.05, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("Seuil IoU (NMS)", 0.1, 1.0, 0.7, 0.05)

    st.divider()
    st.subheader("Source de l'image")
    source = st.radio("Choisir une image", ["Uploader une image", "Exemple aléatoire du test set"])

weights_path = MODELS[model_choice]
if not weights_path.exists():
    st.error(
        f"Poids introuvables : {weights_path}\n\n"
        "Lance d'abord l'entraînement (voir notebooks/finetuning_yolo.ipynb ou src/train.py)."
    )
    st.stop()

model = load_model(str(weights_path))

image = None
if source == "Uploader une image":
    uploaded = st.file_uploader("Image d'une feuille", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
else:
    if not TEST_IMAGES_DIR.exists() or not any(TEST_IMAGES_DIR.iterdir()):
        st.error(f"Aucune image trouvée dans {TEST_IMAGES_DIR}")
    else:
        if st.button("🎲 Tirer une nouvelle image"):
            st.session_state["sample_path"] = random.choice(list(TEST_IMAGES_DIR.glob("*")))
        if "sample_path" not in st.session_state:
            st.session_state["sample_path"] = random.choice(list(TEST_IMAGES_DIR.glob("*")))
        image = Image.open(st.session_state["sample_path"]).convert("RGB")
        st.caption(f"Fichier : {st.session_state['sample_path'].name}")

if image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Image originale")
        st.image(image, use_container_width=True)

    annotated, result = run_detection(model, image, conf_threshold, iou_threshold)

    with col2:
        st.subheader("Détections")
        st.image(annotated, use_container_width=True)

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        st.subheader(f"{len(boxes)} détection(s)")
        rows = []
        for box in boxes:
            cls_id = int(box.cls.item())
            rows.append(
                {
                    "classe": result.names[cls_id],
                    "confiance": round(float(box.conf.item()), 3),
                }
            )
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("Aucune maladie détectée à ce seuil de confiance.")
else:
    st.info("Choisis une image pour lancer la détection.")
