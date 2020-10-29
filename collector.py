import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import argparse
import pickle

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

from util import get_current_datetime, split_data

class CarlaSyncMode(object):
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

def generate_control_dict(control):
    control_dict = {
        'steer': control.steer,
        'throttle': control.throttle,
        'brake': control.brake,
        'hand_brake': control.hand_brake,
        'reverse': control.reverse
    }
    return control_dict

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-r', '--split_ratio',
        default=0.8,
        type=float,
        help='train val split ratio')
    argparser.add_argument(
        '-t', '--time_to_run',
        default=5,
        type=int,
        help='time for data collection vehicle to run')
    argparser.add_argument(
        '-m', '--map',
        default='Town04',
        type=str,
        help='town to drive in')
    argparser.add_argument(
        '-e', '--episodes',
        default=3,
        type=int,
        help='# of epsiodes to run')
    argparser.add_argument(
        '-f', '--frames',
        default=100,
        type=int,
        help='# of frames')
    args =  argparser.parse_args()

    actor_list = []

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()
    world = client.load_world(args.map)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # Get current datetime for versioning
    current_datetime = get_current_datetime()
    
    try:
        m = world.get_map()

        blueprint_library = world.get_blueprint_library()

        vehicles = blueprint_library.filter('vehicle.*')
        cars = [v for v in vehicles if int(v.get_attribute('number_of_wheels')) != 2]
        bp = random.choice(cars)

        vehicle_transform = random.choice(m.get_spawn_points())

        # Spawn test vehicle at start pose
        vehicle = world.spawn_actor(bp, vehicle_transform)

        actor_list.append(vehicle)
        print(f'created my {vehicle.type_id}')

        vehicle.set_autopilot(True)

        # RGB Blueprint
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('enable_postprocess_effects', 'True')

        # CenterRGB
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        center_rgb = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(center_rgb)
        print(f'created center {center_rgb.type_id}')

        # LeftRGB
        camera_transform = carla.Transform(carla.Location(x=1.6, y=-1.25, z=1.7))
        left_rgb = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(left_rgb)
        print(f'created left {left_rgb.type_id}')

        # RightRGB
        camera_transform = carla.Transform(carla.Location(x=1.6, y=1.25, z=1.7))
        right_rgb = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(right_rgb)
        print(f'created right {right_rgb.type_id}')
        

        # Create a synchronous mode context.
        with CarlaSyncMode(world, center_rgb, left_rgb, right_rgb, fps=10) as sync_mode:
            for e in range(args.episodes):
                print(f'starting episode {e:0>4d}')
                episode_label = {}
                for f in range(args.frames):

                    # Advance the simulation and wait for the data.
                    _, center_rgb_img, left_rgb_img, right_rgb_img = sync_mode.tick(timeout=5.0)
                    control_dict = generate_control_dict(vehicle.get_control())
                    print(f, control_dict)

                    # Save Images
                    center_rgb_img.save_to_disk(f'data/{current_datetime}/episode_{e:0>4d}/CenterRGB/{f:06d}.png', carla.ColorConverter.Raw)
                    episode_label[f'episode_{e:0>4d}/CenterRGB/{f:06d}'] = control_dict

                    left_rgb_img.save_to_disk(f'data/{current_datetime}/episode_{e:0>4d}/LeftRGB/{f:06d}.png', carla.ColorConverter.Raw)
                    control_dict['steer'] = 0.25
                    episode_label[f'episode_{e:0>4d}/LeftRGB/{f:06d}'] = control_dict

                    right_rgb_img.save_to_disk(f'data/{current_datetime}/episode_{e:0>4d}/RightRGB/{f:06d}.png', carla.ColorConverter.Raw)
                    control_dict['steer'] = -0.25
                    episode_label[f'episode_{e:0>4d}/RightRGB/{f:06d}'] = control_dict

                print('all frames simulated')

                # Save episode label dict.
                with open(f'data/{current_datetime}/episode_{e:0>4d}/label.pickle', 'wb') as f:
                    pickle.dump(episode_label, f, pickle.HIGHEST_PROTOCOL)

                print('saved episode label dict')

                # Move vehicle to another random spawn point for the next episode
                vehicle.set_transform(random.choice(m.get_spawn_points()))

        # Split episodes into train and val sets
        split_data(f'data/{current_datetime}', args.episodes, args.split_ratio)
    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        # pygame.quit()
        print('done.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
