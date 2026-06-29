# Development Notes

## Main Question

**Can we maintain privacy around the patient head/face region while preserving clinical motion?**

The current video anonymization prototype is still a Level 1 feasibility pipeline. It can blur detected face and high-risk object regions, and the manual temporal tracking logic has improved short detection dropouts. However, recent testing shows that the remaining patient-face failures are mostly caused by detector/coverage limitations rather than only tracking failure.

Key observation from debug review:

- Staff faces are mostly blurred consistently.
- Patient face reveal still happens in short periods, especially when the patient is upward-facing or seen from a clinical angle.
- Increasing face/object hold frames from 15 to 30 reduced some flicker, but did not fully solve patient-face reveal.
- In several failure windows, the tracker held previous patient-face boxes, but the held boxes became stale or did not fully cover the visible face.
- In other windows, YOLO detected the patient face too late, so the tracker had no correct recent box to hold.

Current interpretation:

```text
YOLO face detection is unstable for upward-facing clinical patient poses.
Manual privacy tracking helps, but it cannot protect regions that were never detected or were detected with insufficient coverage.
```

## 1. Face/Head Detection Alternatives

Purpose: reduce patient-face reveal in failure periods.

Test each alternative against the current YOLO face model on the same failure frames/clips:

```text
2:13-2:14
4:21-4:23
4:30-4:31
4:32-4:33
```

Models/tools to try:

- **RetinaFace**: useful benchmark for difficult face angles and partial faces.
- **EgoBlur**: limited to faces/license plates, but worth a small test for upward-facing patient faces.
- **MediaPipe Face Landmarker**: may help when facial landmarks/head pose are partially visible.
- **YOLO face/head variants**: compare alternative YOLO face/head models trained on different datasets.

Specific **EgoBlur** question:

```text
Does EgoBlur detect upward-facing clinical patient faces better than the current YOLO face model?
```

Do not evaluate EgoBlur as a full anonymization pipeline. It does not cover wristbands, screens, documents, tattoos, or other medical privacy risks. If useful, treat it as an additional face detector whose boxes can feed the same temporal privacy tracker.

Expected outcome:

```text
Find whether another face/head detector can detect the patient when current YOLO misses.
```

If yes, possible combination:

```text
YOLO face boxes + RetinaFace/EgoBlur boxes -> same temporal privacy tracker
```

## 2. Specialized YOLO Variants

Purpose: explore whether task-specific YOLO models provide useful privacy or clinical-utility signals.

YOLO variants to explore:

- **YOLO head/face detector**: may cover side, back, or upward-facing head better than face-only detection.
- **YOLO hand detector**: useful when hands cover the patient face; may help preserve clinical hand action while keeping face/head blur active.
- **YOLO pose model**: useful for body/head position and motion cues.
- **YOLO lip/mouth detector**: lower priority, but may be useful if mouth visibility is clinically relevant or identity-sensitive.
- **YOLO screen/document/medical object detector**: useful for Level 2 privacy risks.

Evaluation question for each model:

```text
Does this model provide a useful signal that current YOLO face/object detection misses?
```

Examples:

```text
Hand detector:
  Can it identify hand-face overlap periods?
  Can it help avoid excessive blur over clinical hand motion?

Head detector:
  Can it detect upward-facing or side-facing patient head regions?

Pose model:
  Can it support stable patient/staff localization and future motion-preserving anonymization?

Lip/mouth detector:
  Does it detect partial face regions when full-face detection fails?
```

## 3. Pose, Body, And Landmark Models

Purpose: not directly face blur, but understand body/head/hands and preserve clinical motion.

Models/tools to test:

- **YOLO pose**
- **MediaPipe Pose Landmarker**
- **MediaPipe Hand Landmarker**
- **MediaPipe Face Landmarker**

Expected outcome:

```text
Can pose/head/hand landmarks help define a stable patient region?
Can hands be preserved while patient face/head remains blurred?
Can body or head position inspire a better fallback rule for patient regions?
```

This is especially useful for future Level 2/3 anonymization, but it may also inspire Level 1 fallback rules.

## 4. Object/Region Models

Purpose: improve screens, papers, phones, and other high-risk visual objects.

Models/tools to compare:

- current YOLO object model
- smaller/faster YOLO model for speed comparison
- specialized YOLO screen/document/phone detectors if available
- RF-DETR for future stronger detection/segmentation or fine-tuning

RF-DETR [RF-DETR GitHub](https://github.com/roboflow/rf-detr). is more relevant for future custom healthcare object detection than immediate face blur. It should be evaluated based on:

```text
local deployability
runtime
VRAM/RAM needs
segmentation quality
ability to fine-tune on healthcare-specific objects
```

Expected outcome:

```text
Which model catches screens/documents/phones more consistently, and at what runtime cost?
```

## 5. Custom Healthcare Detector Feasibility

This is likely necessary later because generic YOLO face/object models do not fully match the clinical privacy problem.

The first custom class should not be only "face." A better target is:

```text
patient_head_privacy_region
```

This label can include:

```text
face
forehead
hairline
ears
chin/jaw
partially visible side/upward head region
```

This matches the real privacy target better than a normal face detector box.

Possible first custom classes:

```text
patient_head_privacy_region
screen_or_monitor
phone
paper_or_document
wristband
hand
```

Recommended first training scope:

```text
1-2 classes
100-300 labeled frames
failure-heavy frame selection
train/validation split
compare against current YOLO face model
```

Suggested workflow:

```text
extract frames from failure clips
label boxes
train YOLO
test on held-out clips
compare detection, coverage, and runtime
```

Main training question:

```text
Can a custom YOLO model detect the patient head/privacy region better than generic YOLO face?
```

## 6. Recommended Next Experiment

Create a small comparison set from failure periods:

```text
2:13-2:14
4:21-4:23
4:30-4:31
4:32-4:33
```

Extract:

```text
5-10 frames per period
plus a short video clip around each failure period
```

Compare:

```text
current YOLO face
alternative YOLO face/head models
YOLO hand detector
YOLO pose
RetinaFace
EgoBlur
MediaPipe face/pose/hand landmarks
```

Use a simple comparison table:

```text
Model | Detects patient face/head? | Covers enough region? | Preserves hands? | Runtime | Notes
```

Preferred next order:

```text
1. Test face/head alternatives on failure frames.
2. Test YOLO hand/head/pose variants for useful signals.
3. Start a small YOLO training workflow for patient_head_privacy_region.
4. Use results to decide whether Level 1 needs detector fusion, patient-bed fallback, or a custom model.
```

## 7. Current Detector Comparison Interpretation

Recent model-comparison tests clarified that no single off-the-shelf detector is enough for the clinical privacy problem.

Summary of current findings:

```text
YOLO face:
  Fast and easy to integrate.
  Staff faces are mostly detected well.
  Patient upward-facing face/head region is sometimes detected late, unstable, or too small.

RetinaFace:
  Best face-detection quality so far on difficult patient failure windows.
  Detects the wheeled-in/upward-facing patient earlier than YOLO face.
  Provides useful landmarks: right_eye, left_eye, nose, mouth_right, mouth_left.
  Too slow locally for default full-video processing without optimization or GPU testing.

MediaPipe Face Detector:
  Very fast.
  Short-range and sparse full-range tests were noisy and unstable.
  Threshold tuning reduced some noise but did not solve patient reliability.
  Not a serious detector candidate for this clinical pose problem.

EgoBlur:
  Completed successfully as a standalone anonymization CLI.
  Did not consistently blur the patient face and showed flickering.
  Only blurred facial area rather than the broader patient head/privacy region.
  Missed some angled staff faces that YOLO face detected.
  Detected the wheeled-in patient later than RetinaFace.
  Too slow and less transparent for integration, so not considered further.

YOLO head detector:
  Most promising practical detector so far for broader head/privacy-region coverage.
  Same YOLO output structure as the current face detector, so integration is straightforward.
  Much faster than RetinaFace/EgoBlur.
  Stable in many later patient head windows.
  Still misses or drops briefly when hands/tools touch or occlude the head.
  Detects wheeled-in patient head later than RetinaFace and sometimes later than YOLO face.
```

Current interpretation:

```text
The strongest immediate direction is detector fusion, not replacing one detector with another.
YOLO face and YOLO head fail differently, so combining them should improve privacy coverage.
Tracking/hold logic remains necessary because even the head detector flickers under occlusion.
```

Immediate practical experiment:

```text
YOLO face boxes
+ YOLO head boxes
+ separate face/head temporal tracking states
+ padded blur over the union of tracked boxes
```

This should be tested before spending more time on generic face detectors.

## 8. Clinical Use Vs Privacy Preservation

The current detector work focuses on reliable anonymization:

```text
Prevent visible patient/staff identity leaks.
Keep blur stable enough that short detector dropouts do not reveal identity.
```

However, clinical training value creates a second, later-stage goal:

```text
Preserve clinically meaningful action while still anonymizing identity.
```

This is especially difficult when hands, tools, or tubes interact with the patient face/head. Examples:

```text
hands on or near the patient's face
tools touching the head/face area
intubation where mouth/nose/tool motion may be clinically important
patient motion that should remain visible for training
```

Simple rectangular blur can fail in two ways:

```text
Over-blur:
  Important hand/tool/mouth motion is hidden, reducing clinical usefulness.

Under-blur:
  Hands/tools occlude the face, detector confidence drops, and identity-sensitive head/face regions are briefly revealed.
```

Current-stage decision:

```text
Privacy reliability comes first.
Accept some over-blur while proving the pipeline can prevent identity leaks.
```

Later-stage direction:

```text
Use multi-class detection, segmentation, and/or masks to separate:
  patient_head_privacy_region
  eyes/upper-face identity region
  mouth/nose clinical region
  hands
  tools/instruments
  patient/staff body context
```

Possible future anonymization behavior:

```text
Blur identity-sensitive regions strongly.
Preserve or lightly anonymize clinically important hands/tools where privacy permits.
Use tracking so masks remain stable while hands/tools briefly touch or occlude the face.
```

This question should be recorded as an important product/research direction, but it should not block the current detector-fusion work.

## 9. Human-In-The-Loop Review And Time-Based Tracks

The final product should likely include human review because automatic detectors will not catch every privacy risk.

Basic product flow:

```text
1. User uploads video.
2. System runs automatic anonymization.
3. User reviews the anonymized video.
4. User marks areas that need additional blur.
5. System propagates those manual corrections over time.
6. User reviews the corrected result.
7. Final anonymized video is rendered.
```

Manual corrections should not be stored as only single-frame coordinates. Moving people, heads, hands, tools, and objects need time-aware correction data.

Better representation:

```text
manual_blur_track:
  start_frame
  end_frame
  keyframes:
    frame_number -> box or mask
  tracking_method:
    manual_only | optical_flow | object_tracker | video_segmentation
  blur_policy:
    blur | pixelate | mask | stronger_blur
```

The user can mark one or more keyframes:

```text
draw a rectangle
paint/select a mask
adjust an existing detector track
mark a region to follow forward/backward
```

Then the system should propagate the correction across nearby frames using tracking or video segmentation. The user should only need to correct drift, not annotate every frame.

This suggests that future anonymization code should accept both:

```text
automatic detector tracks
manual user-created blur tracks
```

and render the union of both.

Design implication:

```text
Manual review should be modeled as time-based tracks, not static coordinates.
This is more flexible, more realistic for moving video, and better aligned with final product needs.
```

## 10. Custom Healthcare Detector Feasibility Update

The head-detector results strengthen the case for a custom clinical model.

Generic public detectors do not exactly match the real target. The target is not simply:

```text
face
```

The target is closer to:

```text
clinical patient/staff identity-sensitive head and face region under occlusion, upward pose, hands, tools, and bed movement
```

Long-term training direction:

```text
Fine-tune a YOLO model on clinical failure frames.
Label the privacy region that should actually be blurred, not only anatomical face boxes.
Keep temporal tracking even after fine-tuning, because video detection will still flicker.
```

Possible custom classes:

```text
patient_head_privacy_region
staff_head_privacy_region
face
hand
tool_or_instrument
mouth_nose_clinical_region
screen_or_monitor
paper_or_document
wristband
```

The first custom model should stay small:

```text
Start with patient_head_privacy_region and/or general head_privacy_region.
Use failure-heavy frame selection.
Compare against YOLO face + YOLO head detector fusion.
```

## 11. Recommended Next Experiment

Current recommended order:

```text
1. Implement/test YOLO face + YOLO head fusion with separate tracking states.
2. Test YOLO hand detector as a context signal around face/head occlusion.
3. Test face-parts or body-part detector only if it answers a specific clinical/privacy question.
4. Test pose model after body-part detectors, mainly for patient/staff context and future motion-preserving anonymization.
5. Use all results to decide whether to start custom YOLO training for head/privacy-region labels.
```

Pose should be treated as a context/fallback layer, not the primary blur detector:

```text
head/face detectors -> what to blur
hand/body/pose detectors -> context, occlusion, clinical motion, fallback hints
tracking -> temporal stability
manual tracks -> human-in-the-loop correction
```

## 12. Future Code Structure

The current `src/video.py` script is still useful for prototyping. Future work should move toward a modular video anonymization pipeline.

Initial refactor target:

```text
src/video/
  pipeline.py          # main video processing loop
  config.py            # thresholds, model paths, anonymization levels
  detectors/
    yolo_face.py       # YOLO face detection
    yolo_object.py     # high-risk object detection
  tracking/
    box_tracker.py     # shared IoU-based temporal tracking
  rendering/
    blur.py            # Gaussian blur and future masking/pixelation
```

Longer-term structure:

```text
src/
  video/
    __init__.py

    pipeline.py          # main orchestration
    config.py            # level configs and thresholds
    io.py                # video read/write utilities

    detectors/
      __init__.py
      base.py            # shared Detector interface
      yolo_face.py
      retinaface.py
      yolo_object.py
      pose.py
      manual_regions.py

    tracking/
      __init__.py
      face_tracker.py
      box_tracker.py
      matching.py        # IoU, center distance, track matching

    policies/
      __init__.py
      level1.py          # face/head anonymization
      level2.py          # face + screens/docs/wristbands/etc.
      level3.py          # pose/avatar/skeleton future work

    rendering/
      __init__.py
      blur.py            # Gaussian blur, pixelation, mask
      debug_draw.py      # boxes, labels, track IDs

    evaluation/
      __init__.py
      metrics.py         # missed detections, blur coverage, runtime
      reports.py         # CSV/JSON summaries
```

Design principle:

```text
Detectors find candidate privacy regions.
Trackers stabilize regions across time.
Policies decide what should be anonymized at each level.
Renderers apply blur/mask/pixelation.
Evaluation records whether privacy and clinical utility are preserved.
```
