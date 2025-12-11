# ------------------------------
# Leaf segmentation
# ------------------------------

def _postprocess_leaf_mask(leaf_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[BBox]]:
    """
    Take a binary leaf mask (0/255) and:
      - clean noise
      - split touching leaves with watershed on distance transform
      - return final mask, label image, and bounding boxes
    """
    leaf_mask = (leaf_mask > 0).astype("uint8") * 255

    # Clean up small specks
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_mask, connectivity=8)
    cleaned = np.zeros_like(leaf_mask)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area > 500:
            cleaned[labels == lbl] = 255

    if cv2.countNonZero(cleaned) == 0:
        return cleaned, labels, []

    # Distance transform for splitting
    dist = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # Lower threshold → more peaks → more separation of overlaps
    _, sure_fg = cv2.threshold(dist_norm, 0.25, 1.0, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg * 255)

    kernel = np.ones((5, 5), np.uint8)
    sure_bg = cv2.dilate(cleaned, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed needs 3-channel image
    color_for_ws = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    markers_ws = cv2.watershed(color_for_ws, markers)

    final_mask = np.zeros_like(cleaned)
    final_mask[markers_ws > 1] = 255

    # Label individual leaves
    num_final, labels_final, stats_final, _ = cv2.connectedComponentsWithStats(
        final_mask, connectivity=8
    )

    boxes: List[BBox] = []
    for lbl in range(1, num_final):
        x = stats_final[lbl, cv2.CC_STAT_LEFT]
        y = stats_final[lbl, cv2.CC_STAT_TOP]
        w = stats_final[lbl, cv2.CC_STAT_WIDTH]
        h = stats_final[lbl, cv2.CC_STAT_HEIGHT]
        area = stats_final[lbl, cv2.CC_STAT_AREA]

        if area < 500:
            continue

        boxes.append((x, y, w, h))

    return final_mask, labels_final, boxes


def _segment_leaves_classical(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[BBox]]:
    """
    Classical HSV-based segmentation + watershed, improved to split overlaps.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # A bit narrower and shifted towards “lettuce green”
    # You can tweak these if you change crops.
    lower_green = np.array([30, 25, 40])
    upper_green = np.array([85, 255, 255])

    base_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return _postprocess_leaf_mask(mask)


# -------- Optional: learned model backend -------- #

def _segment_leaves_deep(image_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, List[BBox]]]:
    """
    OPTIONAL: use a trained model for leaf foreground detection.

    This function looks for a file `leaf_model.pth` in the working directory and,
    if PyTorch is available, loads it as a **binary segmentation** UNet-style model
    that returns a single-channel 0-1 mask.

    If anything fails (no torch, no weights, etc.), it returns None and the
    calling code will fall back to the classical pipeline.

    NOTE:
      - You need to provide the trained weights yourself.
      - The simple UNet architecture below must match how you trained it.
    """
    import os

    weights_path = "leaf_model.pth"
    if not os.path.exists(weights_path):
        return None

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception:
        # Torch not installed in this environment
        return None

    # ---- minimal UNet backbone (2-downsample) ---- #
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.down1 = DoubleConv(3, 32)
            self.pool1 = nn.MaxPool2d(2)
            self.down2 = DoubleConv(32, 64)
            self.pool2 = nn.MaxPool2d(2)

            self.bottleneck = DoubleConv(64, 128)

            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.conv2 = DoubleConv(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.conv1 = DoubleConv(64, 32)

            self.out_conv = nn.Conv2d(32, 1, 1)

        def forward(self, x):
            d1 = self.down1(x)
            p1 = self.pool1(d1)
            d2 = self.down2(p1)
            p2 = self.pool2(d2)

            b = self.bottleneck(p2)

            u2 = self.up2(b)
            u2 = torch.cat([u2, d2], dim=1)
            c2 = self.conv2(u2)

            u1 = self.up1(c2)
            u1 = torch.cat([u1, d1], dim=1)
            c1 = self.conv1(u1)

            out = self.out_conv(c1)
            return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet().to(device)
    try:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)
    except Exception:
        # weights not compatible / corrupted etc.
        return None

    model.eval()

    # Prepare image for model (HWC BGR → CHW RGB, 0-1 float)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype("float32") / 255.0
    img = np.transpose(img, (2, 0, 1))  # C,H,W
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        mask = (probs[0, 0].cpu().numpy() > 0.5).astype("uint8") * 255

    return _postprocess_leaf_mask(mask)


def segment_leaves(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[BBox]]:
    """
    High-level segmentation entry point:

      1. Try a **trained deep model** (if leaf_model.pth + torch are available).
      2. If that fails, fall back to the improved classical HSV + watershed method.

    This keeps the rest of your phenotyping code unchanged.
    """
    # First, try the learned model
    try:
        deep_result = _segment_leaves_deep(image_bgr)
    except Exception:
        deep_result = None

    if deep_result is not None:
        return deep_result

    # Fallback: classical pipeline
    return _segment_leaves_classical(image_bgr)
