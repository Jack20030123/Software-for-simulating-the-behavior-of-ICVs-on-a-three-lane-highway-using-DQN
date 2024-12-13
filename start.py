import argparse
import tkinter as tk
from deal import train, test
def ask_vehicle_number():
    global num_vehicles_entry, vehicle_count_entry, root
    root = tk.Tk()
    root.title("控制车辆的数量与非控制车辆的数量")
    tk.Label(root, text="Number of ICVs").pack()
    num_vehicles_entry = tk.Entry(root)
    num_vehicles_entry.insert(0, "3")  # 默认值为3
    num_vehicles_entry.pack()
    tk.Label(root, text="Number of HDVs").pack()
    vehicle_count_entry = tk.Entry(root)
    vehicle_count_entry.insert(0, "10")  # 默认值为10
    vehicle_count_entry.pack()
    submit_button = tk.Button(root, text="Submit", command=create_vehicle_fields)
    submit_button.pack()



def create_vehicle_fields():
    global vehicle_param_entries, root
    root.withdraw()
    num_vehicles = int(num_vehicles_entry.get() or "3")  # 如果没有输入，则默认为3
    vehicle_param_entries = []
    vehicle_window = tk.Toplevel(root)
    vehicle_window.title("Vehicle control parameters")
    canvas = tk.Canvas(vehicle_window)
    scrollbar = tk.Scrollbar(vehicle_window, command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    default_values = {
        'collision_reward_entry': "-1",
        'high_speed_reward_entry': "0.25",
        'headway_reward_entry': "0.25",
        'on_road_reward_entry': "0.01",
        'right_lane_reward_entry': "-0.01",
        'lane_change_reward_entry': "-0.01",
        'acceleration_reward_entry': "0.1",
        'deceleration_reward_entry': "-0.1",
        'speed_entry': "28",
        'initial_lane_id_entry': "2",
        'spacing_entry': "0.6"
    }
    for i in range(num_vehicles):
        entries = {}
        tk.Label(scrollable_frame, text=f"Vehicle {i + 1} parameters").pack()
        for label_text, key in default_values.items():
            tk.Label(scrollable_frame, text=label_text.replace('_entry', '').replace('_', ' ').title()).pack()
            entry = tk.Entry(scrollable_frame)
            entry.insert(0, key)  # 设置默认值
            entry.pack()
            entries[label_text] = entry
        vehicle_param_entries.append(entries)
    submit_button = tk.Button(scrollable_frame, text="submit", command=submit)
    submit_button.pack()
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")



def submit():
    global multi_highway_config
    # 清空或初始化 controlled_vehicles 配置
    multi_highway_config["controlled_vehicles"] = {}

    # 获取并设置非控制车辆的数量
    non_controlled_vehicle_count = int(vehicle_count_entry.get())
    multi_highway_config["vehicles_count"] = non_controlled_vehicle_count

    # 遍历车辆参数条目列表，将每个条目的值存入配置字典中
    for i, entries in enumerate(vehicle_param_entries):
        # 从UI条目中获取数据并转换为适当的数据类型
        collision_reward = float(entries['collision_reward_entry'].get())
        high_speed_reward = float(entries['high_speed_reward_entry'].get())
        headway_reward = float(entries['headway_reward_entry'].get())
        on_road_reward = float(entries['on_road_reward_entry'].get())
        right_lane_reward = float(entries['right_lane_reward_entry'].get())
        lane_change_reward = float(entries['lane_change_reward_entry'].get())
        acceleration_reward = float(entries['acceleration_reward_entry'].get())  # 获取加速奖励
        deceleration_reward = float(entries['deceleration_reward_entry'].get())  # 获取减速奖励
        speed = float(entries['speed_entry'].get())
        initial_lane_id = int(entries['initial_lane_id_entry'].get()) - 1
        spacing = float(entries['spacing_entry'].get())

        # 将每辆车的配置存入全局配置字典
        multi_highway_config["controlled_vehicles"][i] = {
            "collision_reward": collision_reward,
            "high_speed_reward": high_speed_reward,
            "headway_reward": headway_reward,
            "on_road_reward": on_road_reward,
            "right_lane_reward": right_lane_reward,
            "lane_change_reward": lane_change_reward,
            "acceleration_reward": acceleration_reward,  # 保存加速奖励
            "deceleration_reward": deceleration_reward,  # 保存减速奖励
            "speed": speed,
            "initial_lane_id": initial_lane_id,
            "spacing": spacing
        }

    # 关闭root窗口，结束UI会话
    root.quit()



def parse_opt(_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=True, help='True: 训练模式 False: 测试模式 ')
    parser.add_argument('--test_model_dir', type=str, default='runs/train/exp6', help='若当前为测试模式，则会导入该路径文件夹中的模型文件')
    parser.add_argument('--episodes', type=int, default=100000, help='游戏循环的场次')
    parser.add_argument('--hidden_size', type=list, default=[128, 64], help='dqn网络的隐藏层大小')
    parser.add_argument('--batch_size', type=int, default=128, help='每次从经验池中随机抽取batch_size组数据用来训练')
    parser.add_argument('--save_step', type=int, default=100, help='每隔多少轮保存一次模型')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.98, help='对未来的看重')
    parser.add_argument('--memory_capacity', type=int, default=10000, help='经验回放池的大小')
    parser.add_argument('--train_start_memory_capacity', type=int, default=1000, help='经验池多大开始训练')
    parser.add_argument('--copy_step', type=int, default=50, help='dqn每隔多少步将当前网络的参数复制给target网络')
    parser.add_argument('--action_space', type=int, default=5, help='动作空间大小')
    parser.add_argument('--save_path', type=str, default='./runs', help='运行结果文件夹保存路径')
    parser.add_argument('--desc', type=str, default='', help='进度条描述')
    parser.add_argument('--render_mode', type=int, default=0, help='用于展示画面  0:用于在本地展示 1:用于在ipynp中展示 3:不展示')
    parser.add_argument('--exploration_decay', type=float, default=20000, help='随机探索衰减，值越大衰减的越慢')
    parser.add_argument('--min_exploration', type=float, default=0.01, help='最小随机探索概率')
    parser.add_argument('--max_exploration', type=float, default=0.9, help='最大随机探索概率')
    return parser.parse_args(_list)
multi_highway_config = {
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
        },
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {
            "type": "DiscreteMetaAction",
            "min_speed": 25,
            "max_speed": 35,
            "speed_num": 30,
        },
    },
    "lanes_count": 3,
    "speed_limit": 35,
    "vehicles_count": 10,
    "vehicles_controlled": 3,  # 控制车辆数
    "duration": 300,
    "reward_speed_range": [30, 35],
    "distance_between_controlled_vehicles": 15,
    "headway_min_threshold": 10,
    "headway_max_threshold": 20,
    "controlled_vehicles": {},
    "normalize_reward": False,
    "offroad_terminal": True,
    "simulation_frequency": 10,
    "policy_frequency": 2,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,
    "screen_height": 150,
    "centering_position": [0.3, 0.5],
    "scaling": 4,
    "show_trajectories": False,
    "render_agent": True,
    "real_time_rendering": False
}
if __name__ == '__main__':
    ask_vehicle_number()
    root.mainloop()
    opt = parse_opt()
    if opt.train:
        train(opt, multi_highway_config)
    else:
        test(opt)