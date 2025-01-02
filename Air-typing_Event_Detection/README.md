# Environment Configuration

To set up the environment, run the following commands:

```bash
conda create --name detection python=3.8
conda activate detection
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -U openmim
mim install mmengine
mim install mmcv==2.0.0
cd mmdetection
pip install -v -e .
pip install jupyter

```

# Test the Data
The `./data` directory contains the temporal information and heatmaps of finger movements when participants input text fragments. By using these data, running `get_tap_point.ipynb` can identify air-typing events. The `./data` directory also includes manually labeled ground truth. [model weight](https://drive.google.com/file/d/1U42G8ay_qpzLjgo5zrhTMufUff-CsB21/view?usp=sharing) 
