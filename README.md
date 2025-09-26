# Advanced Synthetic Image Augmentation

A compact, dependency‑light toolkit that **generates synthetic images** (simple geometric shapes on varied backgrounds) and then **applies a rich augmentation pipeline** built with OpenCV and PIL. Perfect for bootstrapping CV datasets and stress‑testing training pipelines.

---

## ✨ Features
- **Synthetic data generator** with uniform / gradient / noise backgrounds and random shapes (circle, rectangle/square, triangle) in random sizes, colors, and positions.
- **Composed augmentation pipeline** (deterministic order):
  1) Random rotation  
  2) Horizontal/vertical flip  
  3) Random crop → resize  
  4) Isotropic scale  
  5) Translation  
  6) Brightness & contrast jitter  
  7) HSV (hue/sat/value) shift  
  8) Gaussian noise  
  9) Blur  
  10) Perspective warp  
  11) Cutout  
  12) Morphology (dilate/erode)  
- **Clean Python API** and **CLI** with progress bars.
- **Reproducibility** via `--seed`.

---

## 🧩 Repository contents
```
advanced_synthetic_image_augmentation.py  # main module (generator + pipeline + CLI)
```

> Optional (recommended): add a `requirements.txt` with pinned versions (example further below).

---

## 🚀 Quick start
```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install deps
pip install -U pip
pip install numpy opencv-python pillow tqdm

# 3) Generate 10 base images, each with 5 augmentations
python advanced_synthetic_image_augmentation.py \
  --num-images 10 \
  --augmentations 5 \
  --output-dir out \
  --seed 42

# Output: PNGs in ./out like
#  - img_000_orig.png
#  - img_000_aug_000.png ... img_000_aug_004.png
#  - img_001_orig.png
#  ...
```

---

## 🧪 Python API
```python
from advanced_synthetic_image_augmentation import (
    generate_synthetic_image,
    build_default_pipeline,
)

# Generate one synthetic 256×256 image (PIL.Image)
img = generate_synthetic_image((256, 256))

# Build the default augmentation pipeline and apply
pipeline = build_default_pipeline()
aug = pipeline(img)  # returns PIL.Image
```

### Saving batches
```python
from advanced_synthetic_image_augmentation import save_images

originals = []
aug_lists = []
for _ in range(3):
    base = generate_synthetic_image((256, 256))
    originals.append(base)
    aug_lists.append([pipeline(base) for _ in range(4)])

save_images(originals, aug_lists, "out")
```

---

## ⚙️ CLI usage
The CLI wraps the same API and exposes core knobs:

```bash
python advanced_synthetic_image_augmentation.py \
  --num-images <int>         # base images to generate (default: 10) \
  --augmentations <int>      # augmentations per base image (default: 5) \
  --output-dir <path>        # output folder (default: adv_augmented_images) \
  --seed <int|null>          # optional RNG seed for reproducibility
```

Example:
```bash
python advanced_synthetic_image_augmentation.py \
  --num-images 100 \
  --augmentations 8 \
  --output-dir data/synth_shapes \
  --seed 1337
```

---

## 🛠️ Augmentations & defaults
All transforms are implemented in OpenCV and applied **in order** inside `AugmentationPipeline`.

| Transform | Function | Typical defaults |
|---|---|---|
| Rotation | `random_rotate(img, angle_range=(-45, 45), prob=0.7)` | Random angle; reflect border |
| Flip | `random_flip(img, prob_horizontal=0.5, prob_vertical=0.3)` | Independent H/V coin‑flips |
| Random crop → resize | `random_crop_resize(img, scale_range=(0.7, 0.95), prob=0.6)` | Preserve aspect, then resize back |
| Scale | `random_scale(img, scale_range=(0.8, 1.2), prob=0.5)` | Isotropic |
| Translate | `random_translate(img, translate_range=(-0.1, 0.1), prob=0.5)` | Fraction of width/height |
| Brightness/Contrast | `random_brightness_contrast(img, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3), prob=0.6)` | Scales pixel values |
| HSV Shift | `random_hsv_shift(img, hue_range=(-15, 15), sat_range=(0.7, 1.3), val_range=(0.7, 1.3), prob=0.5)` | Works in HSV space |
| Gaussian Noise | `random_noise(img, sigma_range=(5, 25), prob=0.5)` | Adds noise to channels |
| Blur | `random_blur(img, kernel_size_range=(3, 7), prob=0.4)` | Box/Gaussian based on kernel |
| Perspective | `random_perspective(img, distortion_scale=0.4, prob=0.4)` | Random corner displacements |
| Cutout | `random_cutout(img, max_cutout_size=0.3, prob=0.3)` | Random zeroed rectangle |
| Morphology | `random_morphology(img, prob=0.3)` | Random dilate/erode on gray → merged back |

You can copy the default pipeline and tweak any ranges or probabilities.

```python
from advanced_synthetic_image_augmentation import build_default_pipeline
pipe = build_default_pipeline()
# pipe.transforms is a list of callables; you can replace or reorder items
```

---

## 🖼️ Synthetic backgrounds & shapes
- **Backgrounds:** `uniform` (darkish color), `gradient` (corner‑to‑corner linear), `noise` (grayscale noise replicated to RGB)
- **Shapes:** circle, rectangle/square, triangle; random color, size, and placement; bounds are clipped to the canvas size.

---

## 📦 Requirements
- Python ≥ 3.8
- `numpy`, `opencv-python`, `Pillow`, `tqdm`

**Example `requirements.txt`:**
```
numpy>=1.24
opencv-python>=4.8
Pillow>=10.0
tqdm>=4.66
```

Install with: `pip install -r requirements.txt`

---

## 🔁 Reproducibility
Use `--seed <int>` to seed both `random` and `numpy.random` so generation and augmentations are repeatable.

---

## 🗂️ Output layout
Each base image `i` saves as `img_{i:03d}_orig.png` and its `j`‑th augmentation as `img_{i:03d}_aug_{j:03d}.png` in the chosen output directory.

---

## 🧭 Tips
- Prefer square sizes (e.g., 256×256) for the default transforms.
- If you see black borders after rotation/perspective, that’s expected; the code uses border reflection to minimize artifacts.
- Start with small `augmentations` (e.g., 2–4) and increase once previewed.

---

## 🧱 Roadmap (ideas)
- CLI flags to choose background/shape distributions.
- Save paired masks/labels for segmentation/detection examples.
- Add motion blur, JPEG compression, color temperature, and elastic deformation.
- Multiprocessing for large batches.

---

## 📜 License
Specify your license here (e.g., MIT). Add a `LICENSE` file at the repo root.

---

## 🙌 Acknowledgments
- Built with OpenCV and PIL; inspired by common data augmentation recipes used in modern CV pipelines.

---

## 🔎 Troubleshooting
- **`ModuleNotFoundError: cv2`** → `pip install opencv-python`
- **Weird colors after converting** → ensure you use `cv2.cvtColor` conversions provided (`pil_to_cv` / `cv_to_pil`).
- **No images saved** → check write permissions on `--output-dir` and that disk isn’t full.

