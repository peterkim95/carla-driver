from carla.agent.agent import Agent
from carla.client import VehicleControl


class L5Agent(Agent):
    """
    Simple derivation of Agent Class,
    A trivial agent that goes straight
    """
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
        control = VehicleControl() # TODO: Is this the autopilot control?
        
        # print(control)

        control.throttle = 0.9

        return control
