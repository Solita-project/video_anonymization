# Evaluation Notes

## Test Run - 2026-05-21

Video duration: 6:46
Runtime: 29.37 minutes
Configuration: face detector + high-risk object detector

## Observed Limitations

- Face detection fails when faces are heavily occluded by hands or tools.
- Hair and back-of-head regions are not consistently anonymized by face detection.
- Large blur regions can obscure clinically relevant hand actions.
- Object detection for phone screens is intermittent.

## Future Work

- Evaluate wristband detection approaches.
- Investigate tattoo detection with a custom model.
- Improve fallback detection for masked, occluded, and partially visible faces
