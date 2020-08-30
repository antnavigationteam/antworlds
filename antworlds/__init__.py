from pathlib import Path
from antworlds.common import get_eye_image, get_view_resized

base_path = Path(__file__).absolute().parent.parent

SHADERS_PATH = base_path/"antworlds/engine/shaders"
CUBEMAPS_PATH = base_path/"antworlds/engine/cubemaps"

OMMATIDIA_CACHE_PATH = base_path/"antworlds/insect_eye/cache"

MESHES_PATH = base_path/"data/meshes"

IE_MODELS_PATH = base_path/"data/eyes"
