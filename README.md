# carla-driver

Many scripts are heavily adapted from Carla's own Python API examples. Essentially, I use `collector.py` to collect
the training data (with augmentations) as the car drives around on its own using Carla's default autopilot policy. I 
train the steering neural net with `trainer.py`. Finally, the trained models are put to test in
 `matrix.py`.
