from pathlib import Path
from tensorflow.keras.models import load_model
import bentoml

model_path = Path('data/0.1316-0.9690.h5')

def load_and_save_model(model_path: Path) -> None:
    """Loads a keras model from disk and saves it to BentoML."""
    model = load_model(model_path)
    bento_model = bentoml.keras.save_model("birds_species_identification_model", model)
    print(f"Model saved to path: {bento_model.path}")
    print(f"Model Tag: {bento_model.tag}")

if __name__ == "__main__":
    load_and_save_model(model_path)