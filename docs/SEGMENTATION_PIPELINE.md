# Segmentation-Based Observation Pipeline

Future extension: replace raw grayscale frames with binary segmentation masks as
the PPO policy input. The policy would learn purely from obstacle geometry rather
than texture and lighting, dramatically reducing the sim-to-real gap.

---

## Motivation

The current pipeline feeds 4 stacked grayscale frames (4, 128, 128) to the CNN.
This works well in simulation but generalisation to real forests is uncertain:

- AirSim tree textures look nothing like real trees
- Lighting, shadows, and seasonal variation in the real world are far greater
- The CNN has to learn both "what is a tree" and "how to avoid it" simultaneously

A segmentation mask decouples these two problems:

- **In simulation**: AirSim's built-in segmentation camera provides a perfect
  per-class colour image at zero extra cost — no external model needed
- **At deployment**: a lightweight real-world segmentation model produces the same
  style of binary mask, so the policy input looks identical to what it saw in training

The policy only ever sees binary masks. Sim-to-real transfer reduces to the quality
of the segmentation model, which is a much narrower and better-studied problem.

---

## Architecture

```
Training (AirSim):
  RGB camera  ──────────────────────────────────────────► (discarded)
  Segmentation camera → extract tree pixels → binary mask ┐
                                                           ├─► 4-frame stack → PPO policy
                                                           │   (4, 128, 128) uint8
  Depth camera ──────────────────────────────────────────►  proximity penalty only

Deployment (real drone):
  RGB camera → segmentation model → binary mask ──────────┐
                                                           ├─► 4-frame stack → PPO policy
                                                           │   (same format as training)
  Depth sensor ──────────────────────────────────────────►  proximity penalty only
```

---

## Phase 1 — AirSim Segmentation Camera (Training)

AirSim renders `ImageType.Segmentation` as a flat-colour image where each object
class is assigned a consistent colour based on its mesh/object ID.

### Step 1: Assign segmentation IDs to tree actors

In `settings.json`, tree StaticMeshActors need a unique `SegmentationID` so they
all render as the same colour. Without this they may share the default ID with
other scene objects.

```json
"SegmentationSettings": {
  "InitMethod": "CommonObjectsRandomIDs"
}
```

Alternatively, call the API once at startup to set all tree actors to the same ID:

```python
for name in self.tree_names:
    self.client.simSetSegmentationObjectID(name, 20, is_name_regex=False)
```

Then read back the colour that ID 20 maps to and use it as the tree mask threshold.

### Step 2: Replace `ImageType.Scene` with `ImageType.Segmentation`

In `avoidance_env.py → _get_images()`:

```python
responses = self.client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False),
    airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False),
])

# Extract segmentation image
r = responses[0]
raw = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
n_ch = len(raw) // (r.width * r.height)
seg = raw.reshape(r.height, r.width, n_ch)

# Binarise: tree pixels = 255, everything else = 0
# AirSim encodes object ID as colour; tree ID 20 → look up its RGB value
tree_mask = (seg[:, :, 0] == TREE_R) & (seg[:, :, 1] == TREE_G) & (seg[:, :, 2] == TREE_B)
binary = (tree_mask.astype(np.uint8)) * 255
binary = cv2.resize(binary, (self.img_width, self.img_height))
```

Depth remains unchanged — still fetched separately for the proximity penalty.

### Step 3: Update the observation space

The observation space shape and dtype stay the same `(4, 128, 128) uint8`, so
SB3's CnnPolicy and all callbacks require no changes. Only the image content
changes from grayscale luminance to a binary tree mask.

---

## Phase 2 — Real-World Segmentation Model (Deployment)

### Recommended model: YOLOv8-seg (Ultralytics)

- MIT licensed, free for research and commercial use
- Inference: ~5 ms on GPU, ~30 ms on CPU (suitable for 30 Hz)
- Fine-tuning: ~1–2 hours on a consumer GPU with a few hundred images
- Output: instance segmentation masks, easily combined into a single binary mask

```python
from ultralytics import YOLO

seg_model = YOLO("yolov8n-seg.pt")  # nano variant for speed

def rgb_to_tree_mask(rgb_frame: np.ndarray, img_size: int = 128) -> np.ndarray:
    results = seg_model(rgb_frame, classes=[TREE_CLASS_ID], verbose=False)
    mask = np.zeros((rgb_frame.shape[0], rgb_frame.shape[1]), dtype=np.uint8)
    if results[0].masks is not None:
        for m in results[0].masks.data:
            mask = np.maximum(mask, (m.cpu().numpy() * 255).astype(np.uint8))
    return cv2.resize(mask, (img_size, img_size))
```

### Alternative models

| Model | Size | Notes |
|---|---|---|
| YOLOv8n-seg | 6 MB | Fastest, best for edge deployment |
| YOLOv8s-seg | 22 MB | Better accuracy, still real-time |
| SAM2 (tiny) | 38 MB | Best quality, needs temporal tracking |
| MobileNet-DeepLab | ~10 MB | Good for embedded/onboard compute |

---

## Phase 3 — Fine-tuning the Segmentation Model

A generic YOLOv8-seg trained on COCO does not include a "forest tree from drone
perspective" class. Fine-tuning is needed.

### Data sources

1. **Synthetic labels from AirSim** (free, automatic):
   - Collect RGB + segmentation mask pairs during normal training rollouts
   - AirSim provides perfect pixel-level labels at no cost
   - `simGetImages` with both `Scene` and `Segmentation` in the same call

2. **Real drone footage** (small set):
   - Label 200–500 frames from a real forward-facing drone camera
   - Tools: Roboflow (free tier), CVAT, or Label Studio
   - Existing datasets: search "tree segmentation drone" on Roboflow Universe

3. **Mixed training**:
   - Train predominantly on synthetic (thousands of frames)
   - Fine-tune final layers on real (hundreds of frames)
   - Standard domain adaptation approach

### Fine-tuning YOLOv8-seg

```bash
pip install ultralytics
yolo segment train model=yolov8n-seg.pt data=trees.yaml epochs=50 imgsz=128
```

`trees.yaml`:
```yaml
train: ./data/train/images
val:   ./data/val/images
nc: 1
names: ['tree']
```

---

## Integration with the Existing Codebase

The `ROS2CameraBridge` would need an equivalent segmentation path:

```python
# In ros2_bridge.py — add segmentation topic subscription
seg_topic: str = '/airsim_node/SimpleFlight/front_center_Segmentation/image'

def get_latest_mask(self) -> Optional[np.ndarray]:
    """Return the most recent binary tree mask (H, W) uint8, or None."""
    with self._lock:
        return self._latest_mask.copy() if self._latest_mask is not None else None
```

For deployment without ROS2, the segmentation model runs inline:

```python
if self.seg_model is not None:
    gray = self.seg_model.predict(rgb_frame)   # returns binary mask
else:
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)  # fallback
```

The environment, policy, and all callbacks are unchanged.

---

## Expected Benefits

| | Current (grayscale) | Segmentation masks |
|---|---|---|
| Sim-to-real gap | High (texture/lighting) | Low (binary geometry) |
| Lighting sensitivity | High | None |
| Seasonal variation | High | None |
| Training complexity | Single model | Two models (seg + policy) |
| Inference overhead | None | +5–30 ms/frame |
| Label cost | None | Low (AirSim auto-labels) |

---

## Open Questions

- Does the segmentation mask lose information the policy needs? (e.g. thin branches
  not detected as obstacles but still causing collisions)
- How does the binary mask interact with the proximity penalty, which currently uses
  depth? They could be combined — depth within the mask region only.
- Does the increased frame stack informativeness justify removing depth from the
  penalty and adding it as a masked channel instead?
