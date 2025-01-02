# Environment Configuration

To set up the environment, run the following commands:

```bash
pip install opencv-python numpy pandas mediapipe labelme
```

After recording the video, you can run `plot_heatmap.py` to generate the following:
- Heatmap (required for Air-typing Event Detection)
- Temporal information of fingers during air-typing (required for Air-typing Event Detection)
- Heatmap video (needed for annotating ground truth)

If you want to annotate the ground truth, you can first use `labelme` to annotate the air-typing positions on the heatmap using rectangles. Then, run `anno.py` to annotate the moments when air-typing occurs and the corresponding symbol.

When annotating, press the space key to pause the video. Then, use the left mouse button to select a rectangle. Press the numbers `1` - `3` to set the rectangle type: `1` for a clear air-typing event, `2` for an unclear air-typing event, `3` for finger hovering event. Afterward, input the clicked letter and press the spacebar to continue.
