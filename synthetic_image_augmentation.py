

import argparse
import os
import random
from typing import Callable, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Helper functions for synthetic image generation
# -----------------------------------------------------------------------------
def generate_background(size: Tuple[int, int], bg_type: str = None) -> Image.Image:
    
    width, height = size
    bg_types = ['uniform', 'gradient', 'noise']
    bg_type = bg_type if bg_type in bg_types else random.choice(bg_types)

    if bg_type == 'uniform':
        # Uniform background with a random color
        color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
        return Image.new('RGB', (width, height), color)
    elif bg_type == 'gradient':
        # Create a linear gradient from one corner to another
        start_color = np.array([random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)], dtype=np.float32)
        end_color = np.array([random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)], dtype=np.float32)
        # Create arrays representing a gradient in both directions and combine
        gradient = np.zeros((height, width, 3), dtype=np.float32)
        for i in range(3):
            channel = np.linspace(start_color[i], end_color[i], width, dtype=np.float32)
            gradient[:, :, i] = np.tile(channel, (height, 1))
        gradient = gradient.astype(np.uint8)
        return Image.fromarray(gradient)
    elif bg_type == 'noise':
        # Noise background with random grayscale values
        noise = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        # Convert single channel to RGB by stacking
        noise_rgb = np.stack([noise] * 3, axis=-1)
        return Image.fromarray(noise_rgb)
    else:
        raise ValueError(f"Unsupported background type: {bg_type}")


def generate_synthetic_image(size: Tuple[int, int]) -> Image.Image:
    """Generate a synthetic image with a random shape on a random background.

    Args:
        size: A tuple (width, height) for the image dimensions.

    Returns:
        A PIL Image containing the generated image.
    """
    # Create background
    background = generate_background(size)
    draw = ImageDraw.Draw(background)

    width, height = size

    # Choose a random shape type
    shapes = ['circle', 'square', 'rectangle', 'triangle']
    shape_type = random.choice(shapes)
    
    # Random color for the shape (avoid too dark colors)
    color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

    # Determine shape size and position
    min_dim = min(width, height)
    max_shape_size = int(0.6 * min_dim)
    shape_size = random.randint(int(0.3 * min_dim), max_shape_size)
    x = random.randint(0, width - shape_size)
    y = random.randint(0, height - shape_size)

    if shape_type == 'circle':
        bounding_box = [x, y, x + shape_size, y + shape_size]
        draw.ellipse(bounding_box, fill=color)
    elif shape_type == 'square':
        bounding_box = [x, y, x + shape_size, y + shape_size]
        draw.rectangle(bounding_box, fill=color)
    elif shape_type == 'rectangle':
        rect_width = shape_size
        rect_height = int(shape_size * random.uniform(0.5, 1.5))
        rect_width = min(rect_width, width - x)
        rect_height = min(rect_height, height - y)
        bounding_box = [x, y, x + rect_width, y + rect_height]
        draw.rectangle(bounding_box, fill=color)
    elif shape_type == 'triangle':
        p1 = (x + random.randint(0, shape_size // 4), y)
        p2 = (x + shape_size, y + random.randint(0, shape_size // 4))
        p3 = (x + random.randint(shape_size // 4, shape_size), y + int(shape_size * random.uniform(0.7, 1.3)))
        draw.polygon([p1, p2, p3], fill=color)

    return background


# -----------------------------------------------------------------------------
# Transformation functions for augmentation
# -----------------------------------------------------------------------------
def pil_to_cv(img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an OpenCV BGR image."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv_to_pil(img: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR image to a PIL Image."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def random_rotate(img: np.ndarray, angle_range: Tuple[int, int] = (-45, 45), prob: float = 0.5) -> np.ndarray:
  
    if random.random() < prob:
        angle = random.uniform(*angle_range)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return img


def random_flip(img: np.ndarray, prob_horizontal: float = 0.5, prob_vertical: float = 0.5) -> np.ndarray:
    
    if random.random() < prob_horizontal:
        img = cv2.flip(img, 1)  # Horizontal
    if random.random() < prob_vertical:
        img = cv2.flip(img, 0)  # Vertical
    return img


def random_crop_resize(img: np.ndarray, scale_range: Tuple[float, float] = (0.7, 1.0), prob: float = 0.5) -> np.ndarray:
   
    h, w = img.shape[:2]
    if random.random() < prob:
        scale = random.uniform(*scale_range)
        new_w = int(w * scale)
        new_h = int(h * scale)
        x = random.randint(0, w - new_w)
        y = random.randint(0, h - new_h)
        crop = img[y:y + new_h, x:x + new_w]
        img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def random_scale(img: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2), prob: float = 0.5) -> np.ndarray:
   
    if random.random() < prob:
        h, w = img.shape[:2]
        scale = random.uniform(*scale_range)
        scaled = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        new_h, new_w = scaled.shape[:2]
        # If scaled image is larger, crop center; if smaller, pad with reflection
        if scale >= 1:
            x = (new_w - w) // 2
            y = (new_h - h) // 2
            img = scaled[y:y + h, x:x + w]
        else:
            pad_left = (w - new_w) // 2
            pad_right = w - new_w - pad_left
            pad_top = (h - new_h) // 2
            pad_bottom = h - new_h - pad_top
            img = cv2.copyMakeBorder(scaled, pad_top, pad_bottom, pad_left, pad_right,
                                     borderType=cv2.BORDER_REFLECT_101)
    return img


def random_translate(img: np.ndarray, translate_range: Tuple[float, float] = (-0.2, 0.2), prob: float = 0.5) -> np.ndarray:
    
    if random.random() < prob:
        h, w = img.shape[:2]
        tx = random.uniform(*translate_range) * w
        ty = random.uniform(*translate_range) * h
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return img


def random_brightness_contrast(img: np.ndarray, brightness_range: Tuple[float, float] = (0.8, 1.2),
                               contrast_range: Tuple[float, float] = (0.8, 1.2), prob: float = 0.5) -> np.ndarray:
    
    if random.random() < prob:
        brightness = random.uniform(*brightness_range)
        contrast = random.uniform(*contrast_range)
        img = img.astype(np.float32)
        img = img * contrast + (brightness - 1) * 127
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def random_hsv_shift(img: np.ndarray, hue_range: Tuple[int, int] = (-10, 10),
                     sat_range: Tuple[float, float] = (0.8, 1.2),
                     val_range: Tuple[float, float] = (0.8, 1.2), prob: float = 0.5) -> np.ndarray:
  
    if random.random() < prob:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_shift = random.randint(*hue_range)
        s_scale = random.uniform(*sat_range)
        v_scale = random.uniform(*val_range)
        hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img


def random_noise(img: np.ndarray, sigma_range: Tuple[int, int] = (5, 25), prob: float = 0.5) -> np.ndarray:
    
    if random.random() < prob:
        sigma = random.uniform(*sigma_range)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        img = np.clip(noisy, 0, 255).astype(np.uint8)
    return img


def random_blur(img: np.ndarray, kernel_size_range: Tuple[int, int] = (3, 7), prob: float = 0.5) -> np.ndarray:
    
    if random.random() < prob:
        ksize = random.randrange(kernel_size_range[0], kernel_size_range[1] + 1, 2)
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img


def random_perspective(img: np.ndarray, distortion_scale: float = 0.5, prob: float = 0.3) -> np.ndarray:
    
    if random.random() < prob:
        h, w = img.shape[:2]
        # Original corner points
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        # Randomly shift corners by a fraction of width/height
        max_dx = distortion_scale * w
        max_dy = distortion_scale * h
        dst = np.float32([
            [random.uniform(0, max_dx), random.uniform(0, max_dy)],
            [w - 1 - random.uniform(0, max_dx), random.uniform(0, max_dy)],
            [w - 1 - random.uniform(0, max_dx), h - 1 - random.uniform(0, max_dy)],
            [random.uniform(0, max_dx), h - 1 - random.uniform(0, max_dy)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return img


def random_cutout(img: np.ndarray, max_cutout_size: float = 0.3, prob: float = 0.3) -> np.ndarray:
    
    if random.random() < prob:
        h, w = img.shape[:2]
        cutout_w = random.randint(1, int(w * max_cutout_size))
        cutout_h = random.randint(1, int(h * max_cutout_size))
        x = random.randint(0, w - cutout_w)
        y = random.randint(0, h - cutout_h)
        # Fill cutout with random noise or a constant color (here random noise)
        noise = np.random.randint(0, 256, (cutout_h, cutout_w, 3), dtype=np.uint8)
        img[y:y + cutout_h, x:x + cutout_w] = noise
    return img


def random_morphology(img: np.ndarray, prob: float = 0.3) -> np.ndarray:
   
    if random.random() < prob:
        # Convert to grayscale to apply morphology consistently across channels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel_size = random.choice([3, 5, 7])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        op = random.choice(['dilate', 'erode'])
        if op == 'dilate':
            morph = cv2.dilate(gray, kernel, iterations=1)
        else:
            morph = cv2.erode(gray, kernel, iterations=1)
        # Merge the morphology result back into the color image by replacing the green channel
        img[:, :, 1] = morph
    return img


# -----------------------------------------------------------------------------
# Augmentation pipeline class
# -----------------------------------------------------------------------------
class AugmentationPipeline:
 

    def __init__(self, transforms: List[Callable[[np.ndarray], np.ndarray]]):
        self.transforms = transforms

    def __call__(self, img: Image.Image) -> Image.Image:
        # Convert to OpenCV format
        cv_img = pil_to_cv(img)
        # Apply each transform sequentially
        for t in self.transforms:
            cv_img = t(cv_img)
        # Convert back to PIL
        return cv_to_pil(cv_img)


def build_default_pipeline() -> AugmentationPipeline:
    
    transforms = [
        lambda img: random_rotate(img, angle_range=(-45, 45), prob=0.7),
        lambda img: random_flip(img, prob_horizontal=0.5, prob_vertical=0.3),
        lambda img: random_crop_resize(img, scale_range=(0.7, 0.95), prob=0.6),
        lambda img: random_scale(img, scale_range=(0.8, 1.2), prob=0.5),
        lambda img: random_translate(img, translate_range=(-0.1, 0.1), prob=0.5),
        lambda img: random_brightness_contrast(img, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3), prob=0.6),
        lambda img: random_hsv_shift(img, hue_range=(-15, 15), sat_range=(0.7, 1.3), val_range=(0.7, 1.3), prob=0.5),
        lambda img: random_noise(img, sigma_range=(5, 25), prob=0.5),
        lambda img: random_blur(img, kernel_size_range=(3, 7), prob=0.4),
        lambda img: random_perspective(img, distortion_scale=0.4, prob=0.4),
        lambda img: random_cutout(img, max_cutout_size=0.3, prob=0.3),
        lambda img: random_morphology(img, prob=0.3),
    ]
    return AugmentationPipeline(transforms)


# -----------------------------------------------------------------------------
# Main script logic
# -----------------------------------------------------------------------------
def save_images(originals: List[Image.Image], augmented_lists: List[List[Image.Image]], output_dir: str) -> None:
   
    os.makedirs(output_dir, exist_ok=True)
    for i, orig in enumerate(originals):
        orig.save(os.path.join(output_dir, f"img_{i:03d}_orig.png"))
        for j, aug in enumerate(augmented_lists[i]):
            aug.save(os.path.join(output_dir, f"img_{i:03d}_aug_{j:03d}.png"))


def main(num_images: int, augmentations: int, output_dir: str, seed: int = None) -> None:

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    originals: List[Image.Image] = []
    augmented_lists: List[List[Image.Image]] = []
    img_size = (256, 256)

    pipeline = build_default_pipeline()

    # Generate and augment images with progress bars
    for i in tqdm(range(num_images), desc="Generating images"):
        base = generate_synthetic_image(img_size)
        originals.append(base)
        aug_images: List[Image.Image] = []
        for _ in range(augmentations):
            aug = pipeline(base)
            aug_images.append(aug)
        augmented_lists.append(aug_images)

    save_images(originals, augmented_lists, output_dir)
    print(f"Finished: {num_images} base images and {num_images * augmentations} augmented images saved to '{output_dir}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced synthetic image generation and augmentation.")
    parser.add_argument('--num-images', type=int, default=10, help='Number of base images to generate.')
    parser.add_argument('--augmentations', type=int, default=5, help='Number of augmentations per base image.')
    parser.add_argument('--output-dir', type=str, default='adv_augmented_images', help='Directory for output images.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    args = parser.parse_args()
    main(args.num_images, args.augmentations, args.output_dir, args.seed)