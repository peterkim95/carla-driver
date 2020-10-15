import torch
import torchvision.transforms as transforms
from PIL import Image

from carla.agent.agent import Agent
from carla.client import VehicleControl

from pilotnet import PilotNet, get_transform

class L5Agent(Agent):
    """
    Simple derivation of Agent Class,
    A trivial agent that goes straight
    """
    def __init__(self, net_path='checkpoints/net_best_epoch_20.pt'):
        super().__init__()
        self.pilotnet = PilotNet()
        self.pilotnet.load_state_dict(torch.load(net_path))
        print(f'{net_path} load success')
        self.pilotnet.eval() # set to eval mode

    def run_step(self, measurements, sensor_data, directions, target):
        """
        This function receives the following parameters:

        Measurements: the entire state of the world received by the client from the CARLA Simulator. These measurements contains agent position, orientation, dynamic objects information, etc.
        Sensor Data: The measured data from defined sensors, such as Lidars or RGB cameras.
        Directions: Information from the high level planner. Currently the planner sends a high level command from the following set: STRAIGHT, RIGHT, LEFT, NOTHING.
        Target Position: The position and orientation of the target.
        With all this information, the run_step function is expected to return a vehicle control message, containing: steering value, throttle value, brake value, etc.
        """
        # print(measurements.player_measurements.autopilot_control)
        # print(directions) # TODO: Can we use this directly? Or better to dig deeper into planner?
        # control = VehicleControl()
        # control.throttle = 0.3
        control = self.predict_control(sensor_data)
        # print('predicted: ', control)
        return control

    def predict_control(self, sensor_data):
        rgb_array = sensor_data['CameraRGB'].data.copy()
        # numpy shape should be (H x W x C) in range [0,255] with dtype=np.uint8

        image = Image.fromarray(rgb_array)

        transform = get_transform()

        x = transform(image)
        with torch.no_grad(): # reduce mem usage and speed up computation
            y = self.pilotnet(x.unsqueeze(0)) # TODO: Ew. Do I have to add a batch dimension?
        predicted_steer = y.item()

        control = VehicleControl()
        control.throttle = 0.5
        control.steer = predicted_steer
        return control
