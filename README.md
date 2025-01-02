# AirtypeLogger: How Short Keystrokes in Virtual Space Can Expose Your Semantic Input to Nearby Cameras

**AirtypeLogger** discusses how to infer the corresponding keys based on keystroke positions and temporal order when the input is semantically meaningful English text fragment such as 'fe and soun' (part of 'safe and sound'), the keyboard layout is known, but its size and position are unknown.

This repository contains the official implementation of **AirtypeLogger**. The implementation is divided into three main components:

1. **Preprocessing**: The initial data preparation stage.
2. **Air-Typing Event Detection**: A module responsible for identifying keystroke events.
3. **Keystroke Inference**: A module that infers the keystrokes.

Each component is contained in its own subfolder, with a `README.md` file that provides instructions for training and testing **AirtypeLogger**.
