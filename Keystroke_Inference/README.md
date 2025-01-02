# Environment Configuration

To set up the environment, run the following commands:

```bash
conda create --name AirtypeLogger python=3.8
conda activate AirtypeLogger
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install tensorboard scipy torchmetrics
```

# Test the Data
[model weight](https://drive.google.com/file/d/10Seeposjp-XDi8drfRzatbXrVCYN-AdJ/view?usp=sharing) 

The `./105` directory contains keystroke data generated from inputs involving conversations, emails, and reviews. To obtain the corresponding keystroke inference results, execute the following command:

```bash
python test_105.py
```

The `./angle` directory contains keystroke data from inputs where the text fragments were typed with yaw(β) magntiude of 0, 20, 40, and 60 degrees. To compute the corresponding keystroke inference results, execute:

```bash
python test_angle.py
```

# Training

You can prepare the training dataset by running `./train_data_prepare.py` or download it from the [train dataset](https://drive.google.com/file/d/17ZqNv5u7xDhj7FqIGxgr6c7PP1xRkul3/view?usp=sharing).

Afterwards, run `train.py` to train the keystroke inference module.






