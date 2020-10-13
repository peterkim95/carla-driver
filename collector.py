from __future__ import print_function

import shutil
import argparse
import pickle
import logging
import random
import time
from datetime import datetime

import numpy as np

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

from util import makedirs


def run_carla_client(args):
    number_of_episodes = args.episodes
    frames_per_episode = args.frames
    
    CAMERA_RGB_WIDTH = args.camera_rgb_width
    CAMERA_RGB_HEIGHT = args.camera_rgb_height
    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')

        for episode in range(0, number_of_episodes):
            # Start a new episode.

            if args.settings_filepath is None:

                # Create a CarlaSettings object. This object is a wrapper around
                # the CarlaSettings.ini file. Here we set the configuration we
                # want for the new episode.
                settings = CarlaSettings()
                settings.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=20,
                    NumberOfPedestrians=40,
                    WeatherId=1,
                    # WeatherId=random.choice([1, 3, 7, 8, 14]),
                    QualityLevel=args.quality_level)
                settings.randomize_seeds()

                # Now we want to add a couple of cameras to the player vehicle.
                # We will collect the images produced by these cameras every
                # frame.

                # The center camera captures RGB images of the scene.
                camera_center = Camera('CameraCenterRGB')
                # Set image resolution in pixels.
                camera_center.set_image_size(CAMERA_RGB_WIDTH, CAMERA_RGB_HEIGHT)
                # Set its position relative to the car in meters.
                # TODO: Wish there was a better way to know how these values will actually translate in-game
                camera_center.set_position(0.30, 0, 1.30)
                settings.add_sensor(camera_center)

                # Left RGB camera
                camera_left = Camera('CameraLeftRGB')
                camera_left.set_image_size(CAMERA_RGB_WIDTH, CAMERA_RGB_HEIGHT)
                camera_left.set_position(0.30, -0.75, 1.30)
                settings.add_sensor(camera_left)

                # Right RGB camera
                camera_right = Camera('CameraRightRGB')
                camera_right.set_image_size(CAMERA_RGB_WIDTH, CAMERA_RGB_HEIGHT)
                camera_right.set_position(0.30, 0.75, 1.30)
                settings.add_sensor(camera_right)
                
                # Optional depth camera
                if args.depth:
                    camera1 = Camera('CameraDepth', PostProcessing='Depth')
                    camera1.set_image_size(CAMERA_RGB_WIDTH, CAMERA_RGB_HEIGHT)
                    camera1.set_position(0.30, 0, 1.30)
                    settings.add_sensor(camera1)

                # And maybe a crutch
                if args.lidar:
                    lidar = Lidar('Lidar32')
                    lidar.set_position(0, 0, 2.50)
                    lidar.set_rotation(0, 0, 0)
                    lidar.set(
                        Channels=32,
                        Range=50,
                        PointsPerSecond=100000,
                        RotationFrequency=10,
                        UpperFovLimit=10,
                        LowerFovLimit=-30)
                    settings.add_sensor(lidar)

            else:

                # Alternatively, we can load these settings from a file.
                with open(args.settings_filepath, 'r') as fp:
                    settings = fp.read()

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)

            # Choose one player start at random.
            number_of_player_starts = len(scene.player_start_spots)
            player_start = random.randint(0, max(0, number_of_player_starts - 1))

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode...')
            client.start_episode(player_start)

            
            # Initialize label dict in the episode.
            episode_label = {}

            # Iterate every frame in the episode.
            for frame in range(0, frames_per_episode):

                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()

                # Get autopilot control
                control = measurements.player_measurements.autopilot_control
                # control.steer += random.uniform(-0.1, 0.1)

                # Print some of the measurements.
                print_measurements(measurements)

                # Save the images to disk if requested.
                if args.save_images_to_disk:
                    for name, measurement in sensor_data.items():
                        filename = args.out_filename_format.format(episode, name, frame)
                        measurement.save_to_disk(filename)

                        # Save label
                        label_key = 'episode_{:0>4d}/{:s}/{:0>6d}'.format(episode, name, frame)
                        episode_label[label_key] = generate_control_dict(control)

                # We can access the encoded data of a given image as numpy
                # array using its "data" property. For instance, to get the
                # depth value (normalized) at pixel X, Y
                #
                #     depth_array = sensor_data['CameraDepth'].data
                #     value_at_pixel = depth_array[Y, X]
                #

                # Now we have to send the instructions to control the vehicle.
                # If we are in synchronous mode the server will pause the
                # simulation until we send this control.

                if not args.autopilot:
                    print('sending dummy control')
                    client.send_control(
                        steer=random.uniform(-1.0, 1.0),
                        throttle=0.5,
                        brake=0.0,
                        hand_brake=False,
                        reverse=False)

                else:

                    # Together with the measurements, the server has sent the
                    # control that the in-game autopilot would do this frame. We
                    # can enable autopilot by sending back this control to the
                    # server. We can modify it if wanted, here for instance we
                    # will add some noise to the steer.

                    # control = measurements.player_measurements.autopilot_control
                    # TODO: Does random steering jitter add human-ness?
                    # control.steer += random.uniform(-0.1, 0.1)
                    client.send_control(control)

            # Save episode label dict.
            with open(args.out_labelname_format.format(episode), 'wb') as f:
                pickle.dump(episode_label, f, pickle.HIGHEST_PROTOCOL)


def generate_control_dict(control):
    control_dict = {
        'steer': control.steer,
        'throttle': control.throttle,
        'brake': control.brake,
        'hand_brake': control.hand_brake,
        'reverse': control.reverse
    }
    return control_dict


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.0f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)


def get_current_datetime():
    return datetime.now().strftime('%Y-%m-%d--%H-%M-%S')


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-d', '--depth',
        action='store_true',
        help='enable depth camera')
    argparser.add_argument(
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    argparser.add_argument(
        '-r', '--split_ratio',
        default=0.8,
        type=float,
        help='train val split ratio'
    )
    argparser.add_argument(
        '-e', '--episodes',
        default=3,
        type=int,
        help='# of epsiodes to run'
    )
    argparser.add_argument(
        '-f', '--frames',
        default=300,
        type=int,
        help='# of frames per episode'
    )
    argparser.add_argument(
        '--camera_rgb_width',
        default=800,
        type=int,
        help='width of rgb camera'
    )
    argparser.add_argument(
        '--camera_rgb_height',
        default=600,
        type=int,
        help='height of rgb camera'
    )
    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    current_datetime = get_current_datetime()
    args.out_filename_format = 'data/' + current_datetime + '/episode_{:0>4d}/{:s}/{:0>6d}'
    args.out_labelname_format = 'data/' + current_datetime + '/episode_{:0>4d}/label.pickle'

    while True:
        try:

            run_carla_client(args)
            print('Finished simulation.')

            split_data(f'data/{current_datetime}', args.episodes, args.split_ratio)
            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)

def split_data(data_path, max_episodes, split_ratio):
    makedirs(f'{data_path}/train')
    makedirs(f'{data_path}/val')
    
    episodes = np.arange(max_episodes)
    np.random.shuffle(episodes)
    idx = int(max_episodes * split_ratio)
    train, val = episodes[:idx], episodes[idx:]

    for e in train:
        shutil.move(f'{data_path}/episode_{e:0>4d}', f'{data_path}/train/episode_{e:0>4d}')
    for e in val:
        shutil.move(f'{data_path}/episode_{e:0>4d}', f'{data_path}/val/episode_{e:0>4d}')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
