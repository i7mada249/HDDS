# Research Paper: Vision Module For The Hybrid Drone Detection System

## Abstract

This paper documents the vision module used in the Hybrid Drone Detection
System (HDDS), explains the object-detection algorithm that was adopted,
defends the engineering choices behind it, and records the training outputs
that remain in the current repository.

A key repository fact is that the cleaned package `Project v1/` does not
currently contain a standalone `src/vision/` module or a local
`Project v1/notebooks/` directory. The visual branch is instead represented by:

1. the live `yolov5/` tree at the repository root,
2. trained weight files such as `best.pt`,
3. training run folders under `yolov5/runs/train/`,
4. the integrated multimodal application in `App/main.py`.

Therefore, this report is grounded in the current `yolov5/` evidence and the
actual integration code that runs the detector today.

## 1. Repository Evidence Reviewed

The main vision-related evidence currently present is:

- `/home/mo/dev/python/HDDS2/yolov5/`
- `/home/mo/dev/python/HDDS2/yolov5/best.pt`
- `/home/mo/dev/python/HDDS2/yolov5/yolov5s.pt`
- `/home/mo/dev/python/HDDS2/yolov5/yolov5n6.pt`
- `/home/mo/dev/python/HDDS2/yolov5/runs/train/exp/`
- `/home/mo/dev/python/HDDS2/yolov5/runs/train/exp2/`
- `/home/mo/dev/python/HDDS2/App/main.py`

Important cleanup note:

- `Project v1/notebooks/` is not present in the cleaned tree.
- The current `Project v1` package does not include a packaged vision module.
- Vision remains primarily an integration-layer branch in the app.

This fact matters because the vision module must be described from preserved
training/configuration artifacts and the live runtime code, not from a clean
`Project v1/src/vision/` package.

## 2. Problem Definition

The purpose of the vision branch is to detect drones in video frames and
provide a binary detection signal plus localization boxes for the multimodal
dashboard.

Unlike the sound branch, the vision branch is not only about classification. It
must answer two questions simultaneously:

1. Is a drone present in this frame?
2. If yes, where is it located?

This is why a pure image classifier would be insufficient. The project needs an
object detector, not just a frame-level class label.

## 3. Why YOLOv5 Was Used

The detector family used in the repository is YOLOv5. This is shown by the
live `yolov5/` project tree and by the runtime loader in `App/main.py`.

The app loads the model with:

```python
self.yolo_model = torch.hub.load(
    str(YOLO_DIR),
    "custom",
    path=str(YOLO_WEIGHTS),
    source="local",
    device="cpu",
)
```

This is a strong clue that the system uses a custom-trained YOLOv5 detection
model rather than a generic pretrained checkpoint only.

YOLOv5 is a defendable choice for this project because:

- it is an object detector, not only a classifier,
- it is fast enough for frame-by-frame inference,
- it is widely used and well understood,
- it supports transfer learning from pretrained weights,
- it provides both confidence scores and bounding boxes,
- it integrates well with Python-based deployment code.

For a drone-detection thesis, this is the correct detector family to choose
over slower two-stage detectors when the project also needs practical demo
playback and multimodal synchronization.

## 4. Runtime Vision Pipeline In The App

The integrated application keeps the vision branch inside `App/main.py`. The
runtime constants are:

```python
VISION_CONFIDENCE = 0.5
VISION_EVERY_N_FRAMES = 3
```

This means:

- detections below `0.5` confidence are rejected,
- inference is run every third frame unless playback synchronization forces a
  refresh.

This is not an arbitrary optimization. It is a deliberate tradeoff:

- lower compute cost,
- smoother synchronized playback,
- acceptable temporal granularity for a dashboard-style demonstration.

The detector itself is applied as follows:

```python
def _run_vision(self, frame: np.ndarray) -> None:
    self.last_vision_boxes = []
    self.last_vision_detected = False
    if self.yolo_model is None:
        return
    results = self.yolo_model(frame)
    for *xyxy, conf, _cls in results.xyxy[0]:
        confidence = float(conf)
        if confidence < VISION_CONFIDENCE:
            continue
        x1, y1, x2, y2 = [int(value) for value in xyxy]
        self.last_vision_boxes.append((x1, y1, x2, y2, confidence))
    self.last_vision_detected = bool(self.last_vision_boxes)
```

This excerpt is the practical heart of the current vision module. It shows the
complete online decision path:

1. run YOLO inference on the frame,
2. threshold the detections by confidence,
3. convert coordinates into integer bounding boxes,
4. store the accepted detections,
5. derive a binary decision from the presence of at least one valid box.

For multimodal fusion, this is entirely appropriate. The dashboard does not
need dense detector internals at every stage; it needs a stable per-frame
decision and interpretable boxes.

## 5. Visualization And Interpretability

The accepted boxes are drawn back onto the frame:

```python
for x1, y1, x2, y2, confidence in self.last_vision_boxes:
    cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 197, 94), 2)
    cv2.putText(
        frame,
        f"drone {confidence:.2f}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (34, 197, 94),
        2,
    )
```

This matters from a research-defense perspective because visual localization is
immediately interpretable to supervisors and teammates. It is far easier to
trust and debug a detector when the predicted box is visible on the frame.

## 6. Preserved YOLO Training Configuration

The current repository still contains two preserved YOLO training runs:

- `yolov5/runs/train/exp/`
- `yolov5/runs/train/exp2/`

The first run used:

```yaml
weights: yolov5s.pt
data: ../drone.yaml
epochs: 50
batch_size: 16
imgsz: 640
optimizer: SGD
```

The second run used:

```yaml
weights: yolov5n6.pt
data: ../drone.yaml
epochs: 50
batch_size: 16
imgsz: 640
optimizer: SGD
```

Both runs also preserve the same augmentation and optimization hyperparameters,
including:

```yaml
lr0: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
translate: 0.1
scale: 0.5
fliplr: 0.5
mosaic: 1.0
```

This training record is valuable because it tells us the vision branch was not
just a one-click inference demo. There were at least two actual custom training
attempts with different pretrained starting points.

## 7. Defense Of The Vision Training Strategy

### 7.1 Transfer Learning From Pretrained YOLO Weights

The two runs start from:

- `yolov5s.pt`
- `yolov5n6.pt`

This is a correct strategy for a drone-detection project because:

- the dataset is usually smaller than COCO-scale datasets,
- transfer learning reduces training time,
- pretrained backbones provide generic visual features,
- fine-tuning can adapt those features to drone imagery.

The use of two starting checkpoints also shows experimentation:

- `yolov5s.pt` provides a conventional small YOLOv5 model,
- `yolov5n6.pt` provides a lighter architecture with a different scaling
  profile.

This suggests the project was exploring the tradeoff between efficiency and
accuracy.

### 7.2 Why Object Detection, Not Simple Classification

Drone scenes often contain:

- sky background,
- cluttered urban context,
- birds,
- small moving objects,
- scale variation.

A classifier operating on whole frames would be weak because it would not force
the model to localize the object. YOLO-based detection is therefore preferable
because it predicts both object presence and location.

### 7.3 Why The Hyperparameters Are Reasonable

The preserved YOLO hyperparameters are consistent with standard fine-tuning:

- `epochs: 50` is moderate for transfer learning,
- `batch_size: 16` is practical for common GPU limits,
- `imgsz: 640` is the standard YOLOv5 operating size,
- `optimizer: SGD` is a stable default for YOLOv5 training,
- `mosaic: 1.0`, `scale: 0.5`, and `fliplr: 0.5` help generalization.

These choices are easy to defend because they are neither reckless nor exotic.
They align with normal YOLOv5 training practice.

## 8. Training Outputs Preserved In The Repository

The following vision-training outputs are still preserved:

- `yolov5/best.pt`
- `yolov5/yolov5s.pt`
- `yolov5/yolov5n6.pt`
- `yolov5/runs/train/exp/labels.jpg`
- `yolov5/runs/train/exp/labels_correlogram.jpg`
- `yolov5/runs/train/exp/train_batch0.jpg`
- `yolov5/runs/train/exp/train_batch1.jpg`
- `yolov5/runs/train/exp/train_batch2.jpg`
- `yolov5/runs/train/exp/opt.yaml`
- `yolov5/runs/train/exp/hyp.yaml`
- corresponding files for `exp2`

Current artifact sizes of interest are:

- `best.pt`: about `6.3 MB`
- `yolov5s.pt`: about `15 MB`
- `yolov5n6.pt`: about `6.9 MB`

What these outputs prove:

- the dataset was loaded into YOLO training runs,
- label distributions were visualized,
- augmented training batches were inspected,
- at least one best-performing checkpoint was saved.

What is not preserved:

- `results.csv`
- precision-recall plots,
- F1 curves,
- confusion-matrix images,
- validation metric tables,
- the referenced dataset file `../drone.yaml`

That last point is especially important. The `opt.yaml` files reference
`../drone.yaml`, but that file is not present in the current `yolov5/` tree.
Therefore, the training run configuration remains visible, but the full dataset
binding is not completely preserved in-place.

## 9. Inference Behavior In The Current System

The current app uses the trained weights through local YOLO loading. It also
couples inference to video playback timing:

```python
elapsed_s = max(0.0, time.monotonic() - self.playback_started_at)
target_frame = int(elapsed_s * self.video_fps)
if target_frame > self.frame_index + 1:
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    self.frame_index = target_frame
```

This is a practical design choice because it prevents the dashboard from
drifting badly if inference or rendering becomes slower than nominal frame
rate.

The video is also resized before detection overlays are shown:

```python
frame = cv2.resize(frame, (800, 500))
```

This is reasonable for a fixed-layout GUI, though it means visual appearance in
the app is standardized for presentation rather than raw dataset resolution.

## 10. Role Of The Vision Module In Multimodal Fusion

The visual detector provides:

- localization boxes,
- confidence-filtered frame-level detection,
- a boolean signal for fusion.

In the app, the multimodal decision is:

```python
positives = sum([self.last_vision_detected, audio_detected, radar_detected])
if positives >= 2:
    fusion_text, fusion_color = "Confirmed", RED
elif positives == 1:
    fusion_text, fusion_color = "Watch", AMBER
else:
    fusion_text, fusion_color = "Clear", GREEN
```

This means the vision branch contributes one vote toward confirmation. That is
exactly the correct role for computer vision in this hybrid system:

- strong when the object is visible,
- weaker in darkness or blur,
- complementary to sound and radar.

## 11. Strengths And Limitations

Strengths:

- uses a real object detector rather than ad hoc heuristics,
- supports localization and explainability,
- based on transfer learning from proven YOLO checkpoints,
- preserves evidence of at least two training runs,
- integrates directly into the multimodal app.

Limitations:

- not yet extracted into a clean `Project v1/src/vision` package,
- no preserved metric CSV or PR/F1 plots in the current tree,
- referenced dataset YAML is missing from live `yolov5/`,
- current runtime is CPU-based in the app,
- binary frame decision is based on any surviving box above threshold.

## 12. Final Defense Of The Vision Module

The vision module is a valid and technically defendable branch of the project.
Its use of YOLOv5 is appropriate for the drone-detection problem because the
task requires localization, robustness, and practical speed. The training
artifacts that remain show that the team did not simply download a detector and
stop; it ran custom training experiments from at least two pretrained starting
points, preserved the final best checkpoint, and kept representative run
artifacts.

The main weakness is not the algorithm choice. The main weakness is packaging:
the visual logic remains embedded in the application layer and the full metric
history of training was not retained after cleanup.

Even with that limitation, the evidence is strong enough to say that the vision
module is real, custom, trained, and operational in the current HDDS pipeline.
