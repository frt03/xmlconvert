# xmlconvert

1. Convert [metamorph](https://github.com/agrimgupta92/metamorph)'s xml files to Brax's `_SYSTEM_CONFIG`.
```bash
python mujoco_converter.py --xml_model_path ./xml/floor-1409-1-6-01-07-43-16.xml  --config_path ./xml/floor-1409-1-6-01-07-43-16.txt --ignore_unsupported_joints
python mujoco_converter.py --xml_model_path ./xml/floor-1409-9-9-01-14-34-07.xml  --config_path ./xml/floor-1409-9-9-01-14-34-07.txt --ignore_unsupported_joints
python mujoco_converter.py --xml_model_path ./xml/floor-5506-10-1-01-12-44-00.xml  --config_path ./xml/floor-5506-10-1-01-12-44-00.txt --ignore_unsupported_joints
```

2. Run PPO (forward & goal reaching)
```bash
python train_ppo.py
```


- [metamorph](https://github.com/agrimgupta92/metamorph)
- XML filies are available:
```
# Install gdown
pip install gdown
# Download data
gdown 1LyKYTCevnqWrDle1LTBMlBF58RmCjSzM
# Unzip
unzip unimals_100.zip
```
