import random
from typing import Optional, TypeVar

import numpy as np
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env import utils
Observation = TypeVar("Observation")
default_config = {
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",  # 每个控制车辆的状态信息类型
            "vehicles_count": 12,  # 观测到的车辆最大数目
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
        },
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
            "min_speed": 20,  # 车辆最小速度
            "max_speed": 60,  # 车辆最大速度 不会超过车道最大速度
            "speed_num": 10,  # 速度离散化数目  min_speed - max_speed 之间离散化speed_num个速度
        },
    },
    "lanes_count": 4,  # 高速公路的车道
    "speed_limit": 40, # 车道速度限制
    "vehicles_count": 20,  # 非控制车辆数目
    "vehicles_controlled": 3,  # 控制车辆数
    "duration": 30,  # 仿真时长 [s]  不是真实时长
    "reward_speed_range": [20, 40],  # 该速度范围才有速度奖励 超过最大值奖励达到最大
    "distance_between_controlled_vehicles": 10,  # 控制车辆之间的距离阈值m，阈值之内车辆奖励替换为平均奖励
    "headway_min_threshold": 50,  # 车头间距最小阈值
    "headway_max_threshold": 70,  # 车头间距最大阈值
    "avg_speed_reward_min_distance": 50,  # 接收平均速度奖励的最小距离阈值
    "avg_speed_reward_max_distance": 70,  # 接收平均速度奖励的最大距离阈值
    # 受控制的车辆数目以及配置
    "controlled_vehicles": {
        0: {
            "speed": 20,#初始速度
            "initial_lane_id": 0,
            "spacing": 0.36,  # 位置
            # =========奖励相关参数============
            "collision_reward": -1,  # 碰撞奖励，与碰撞时收到的奖励相关
            "high_speed_reward": 0.4,  # 全速行驶奖励，根据config["reward_speed_range"]，低速时线性映射到0
            "headway_reward": 10,     # 车头间距奖励值
            "on_road_reward": 0.1,  # 在路上的奖励，与在路上时收到的奖励相关
            "right_lane_reward": 0,  # 右侧车道奖励，其他车道线性映射到0
            "lane_change_reward": 0,  # 每次变道奖励
            "acceleration_reward": 0.2,  # 加速奖励
            "deceleration_reward": -0.2,  # 减速奖励
        },
        1: {
            "speed": 20, #初始速度
            "initial_lane_id": 1,
            "spacing": 0.30,  # 位置
            # =========奖励相关参数============
            "collision_reward": -1,  # 碰撞奖励，与碰撞时收到的奖励相关
            "high_speed_reward": 0.4,  # 全速行驶奖励，根据config["reward_speed_range"]，低速时线性映射到0
            "headway_reward": 10,     # 车头间距奖励值
            "on_road_reward": 0.1,  # 在路上的奖励，与在路上时收到的奖励相关
            "right_lane_reward": 0,  # 右侧车道奖励，其他车道线性映射到0
            "lane_change_reward": 0,  # 每次变道奖励
            "acceleration_reward": 0.2,  # 加速奖励
            "deceleration_reward": -0.2,  # 减速奖励
        }
    },
    "normalize_reward": True,  # 是否对奖励进行归一化
    "offroad_terminal": True,  # 是否在离开路面时终止仿真
    "simulation_frequency": 15,  # 仿真频率，每秒进行15次仿真步骤 [Hz]
    "policy_frequency": 1,  # 策略更新频率，每秒进行1次策略更新 [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",  # 其他车辆的类型，使用IDM模型
    "screen_width": 600,  # 屏幕宽度，用于图形渲染 [px]
    "screen_height": 150,  # 屏幕高度，用于图形渲染 [px]
    "centering_position": [0.3, 0.5],  # 屏幕中心位置的归一化坐标，x坐标为0.3，y坐标为0.5
    "scaling": 4,  # 屏幕缩放因子，用于图形渲染
    "show_trajectories": False,  # 是否显示车辆轨迹
    "render_agent": True,  # 是否渲染代理车辆
    "real_time_rendering": False  # 是否实时渲染
}


def get_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


class Multi_Highway_Env(AbstractEnv):
    def __init__(self, config: dict = None, render_mode: Optional[str] = None):
        super().__init__(config, render_mode)
        # 设置所有车辆的最大速度限制
        Vehicle.MAX_SPEED = config.get("speed_limit",60)
        # 设置车辆速度限制
        self.config["action"]["action_config"]["target_speeds"] = np.linspace(
            config["action"]["action_config"].get("min_speed", 20),
            config["action"]["action_config"].get("max_speed", 40),
            config["action"]["action_config"].get("speed_num", 3))

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=self.config.get("speed_limit",60)),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    # 创建车辆
    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        others = self.config["vehicles_count"]
        forward_num = int(others / 3)  # 均分非控制车辆到前后
        self.controlled_vehicles = []
        # 创建前方非控制车辆
        for _ in range(forward_num):
            lane_id = self.np_random.integers(0, self.config["lanes_count"] - 1)  # 选择除最右侧车道外的车道
            vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_controlled"],
                                                        lane_id=lane_id)
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)
        # 创建控制车辆
        for id, cfg in self.config["controlled_vehicles"].items():
            vehicle = Vehicle.create_random(
                self.road,
                speed=cfg["speed"],
                lane_id=cfg["initial_lane_id"],
                spacing=cfg["spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)
        # 创建后方非控制车辆
        for _ in range(others - forward_num):
            lane_id = self.np_random.integers(0, self.config["lanes_count"] - 1)  # 选择除最右侧车道外的车道
            vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_controlled"],
                                                        lane_id=lane_id)
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

    def _calculate_headway(self, vehicle):
        min_headway = float('inf')
        for other_vehicle in self.controlled_vehicles:
            if vehicle is not other_vehicle and vehicle.lane_index == other_vehicle.lane_index:
                headway = other_vehicle.position[0] - vehicle.position[0]
                if 0 < headway < min_headway:
                    min_headway = headway
        return min_headway

    def get_multi_avg_speed(self, vehicle_id, _rewards):
        '''
        获得受控制车辆和附近受控制车辆的vehicle_id的平均速度
        :param vehicle_id: 当前受控车辆id
        :param _rewards: 所有受控车辆的奖励信息
        :return: 受控车辆和附近受控车辆的vehicle_id的平均速度
        '''
        sum_speed = [_rewards[vehicle_id].get("high_speed_reward", 0)]
        # 读取新的距离阈值配置
        min_distance = self.config.get("avg_speed_reward_min_distance", 50)
        max_distance = self.config.get("avg_speed_reward_max_distance", 70)
        for idx, reward in _rewards.items():
            if idx == vehicle_id:
                continue
            distance = get_distance(self.controlled_vehicles[vehicle_id].position,
                                    self.controlled_vehicles[idx].position)
            # 如果距离在新指定的范围内，将速度加入到sum_speed中
            if min_distance <= distance <= max_distance:
                sum_speed.append(reward.get("high_speed_reward", 0))
        return sum(sum_speed) / len(sum_speed) if sum_speed else 0

    def _reward(self, action: Action):
        multi_rewards = {}
        _rewards = self._rewards(action)  # 获取每个车辆基于当前状态的奖励信息

        for vehicle_id, rewards in _rewards.items():
            vehicle_config = self.config["controlled_vehicles"][vehicle_id]

            # 检查动作并应用加速或减速奖励
            acceleration_reward = 0
            if action[vehicle_id] == 3:  # 动作3对应加速
                acceleration_reward = vehicle_config.get("acceleration_reward", 0)
            elif action[vehicle_id] == 4:  # 动作4对应减速
                acceleration_reward = vehicle_config.get("deceleration_reward", 0)

            # 计算其他类型的奖励
            for name, value in rewards.items():
                if name == "high_speed_reward":
                    # 如果车辆处于高速状态，则可能需要根据周围的车辆速度调整高速奖励
                    value = self.get_multi_avg_speed(vehicle_id, _rewards)
                rewards[name] = value * vehicle_config.get(name, 0)

            # 计算车头间距奖励
            headway_reward = 0
            headway = self._calculate_headway(self.controlled_vehicles[vehicle_id])
            if self.config["headway_min_threshold"] <= headway <= self.config["headway_max_threshold"]:
                headway_reward = vehicle_config.get("headway_reward", 0)

            # 总奖励是各部分奖励之和
            total_reward = sum(rewards.values()) + headway_reward + acceleration_reward

            # 如果设置了奖励归一化
            if self.config["normalize_reward"]:
                total_reward = utils.lmap(total_reward,
                                          [vehicle_config.get("collision_reward", 0),
                                           vehicle_config.get("high_speed_reward", 0) + vehicle_config.get(
                                               "right_lane_reward", 0) + headway_reward],
                                          [0, 1])

            # 应用在路上的奖励因子
            total_reward *= rewards.get('on_road_reward', 1)
            multi_rewards[vehicle_id] = total_reward

        return multi_rewards

    def _info(self, obs: Observation, action: Optional[Action] = None) -> dict:
        """
        Return a dictionary of additional information

        :param obs: current observation
        :param action: current action
        :return: info dict
        """
        info = {
            "speed": [vehicle.speed for vehicle in self.controlled_vehicles],
            "crashed": [vehicle.crashed for vehicle in self.controlled_vehicles],
            "action": action,
        }
        try:
            info["rewards"] = self._rewards(action)
        except NotImplementedError:
            pass
        return info

    def multi_rewards_func(self, control_vehicle: ControlledVehicle):
        '''
        获得受控制车辆的奖励信息
        :param control_vehicle: 受控制车辆
        :return: 受控制车辆的奖励信息
        '''
        neighbours = self.road.network.all_side_lanes(control_vehicle.lane_index)
        lane = control_vehicle.target_lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = control_vehicle.speed * np.cos(control_vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(control_vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(control_vehicle.on_road)
        }

    def _rewards(self, action: Action):
        rewards = {}
        for vehicle_id, controlled_vehicle in enumerate(self.controlled_vehicles):
            rewards[vehicle_id] = self.multi_rewards_func(controlled_vehicle)
        return rewards

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    def _is_terminated(self) -> bool:
        for controlled_vehicle in self.controlled_vehicles:
            if controlled_vehicle.crashed:
                return True
            if self.config["offroad_terminal"] and not controlled_vehicle.on_road:
                return True
        return False


if __name__ == '__main__':
    env = Multi_Highway_Env(default_config)
    obs = env.reset()
    control_vehicles = len(env.controlled_vehicles)
    eposides = 10
    rewards = [0 for _ in range(control_vehicles)]
    # 0: 'LANE_LEFT',
    # 1: 'IDLE',
    # 2: 'LANE_RIGHT',
    # 3: 'FASTER',
    # 4: 'SLOWER'
    print(env.action_space)
    for eq in range(eposides):
        obs = env.reset()
        # print(obs)
        env.render()
        done = False
        truncated = False
        while not done and not truncated:
            # action = env.action_space.sample()
            # action1 = random.sample([0,1,2,3,4], 1)[0]
            # action2 = random.sample([0,1,2,3,4], 1)[0]
            action = tuple([random.sample([0, 1, 2, 3, 4], 1)[0] for _ in range(control_vehicles)])
            obs, reward, done, truncated, info = env.step((3, 3))
            env.render()
            for i in range(control_vehicles):
                rewards[i] += reward[i]
            control_vehicle = env.controlled_vehicles[0]  # type:ControlledVehicle
            print(info)
            # print(control_vehicle.speed)
            # print(env.controlled_vehicles[1].speed)
            # print("1\n")
            # for b in obs:
            #     print(b)
            print(reward)
            print(done)

            # print(obs)
        # print(rewards)
