import os
import numpy as np
from IPython import display
from matplotlib import pyplot as plt
from tqdm import tqdm
from multi_highway_env import Multi_Highway_Env
from dqn import DQN, MODELS_DIR
from tools import get_log_path, logger_setting, print_args, save_config, load_config, PlotUtil
from tools import save_to_excel
action2action = {0: "lane_left", 1: 'idle', 2: 'lane_right', 3: 'faster', 4: 'slower'}
def train(opt, multi_highway_config):
    opt.input_size = len(multi_highway_config["observation"]["observation_config"]["features"]) * \
                     multi_highway_config["observation"]["observation_config"]["vehicles_count"]
    opt.output_size = opt.action_space
    save_dir = get_log_path(os.path.join(opt.save_path, "train"))
    opt.save_dir = save_dir
    opt.num_agent = len(multi_highway_config['controlled_vehicles'])
    logger = logger_setting(save_dir)
    print_args(logger, opt, multi_highway_config)
    save_config(os.path.join(save_dir, MODELS_DIR), opt, multi_highway_config)
    env = Multi_Highway_Env(multi_highway_config)
    count = 0
    each_avg_reward = {}
    each_total_reward = {}
    total_total_rewards = []
    avg_total_rewards = []
    total_avg_rewards = []
    avg_avg_rewards = []
    avg_speeds = []
    crasheds = []
    crashed_num=0
    steps = []
    ep_list = []
    plot_thread = PlotUtil()
    for i in range(opt.num_agent):
        each_avg_reward[i] = []
        each_total_reward[i] = []
        avg_speeds.append([])
    dqn_agent_list = [DQN(opt, str(i)) for i in range(opt.num_agent)]
    if opt.render_mode == 1:
        img = plt.imshow(env.render(mode='rgb_array'))
    logger.info("填充经验池")
    while len(dqn_agent_list[0].memory) < opt.train_start_memory_capacity:
        print(len(dqn_agent_list[0].memory), "/", opt.train_start_memory_capacity)
        s = env.reset()[0]
        truncated = False
        done = False
        while not truncated and not done:
            a_list = []
            for agent_id, dqn in enumerate(dqn_agent_list):
                a, is_random = dqn.choose_action(s[agent_id], 1)
                a_list.append((a, is_random))
            s_, r, done, truncated, info = env.step(tuple(map(lambda x: x[0], a_list)))
            if opt.render_mode == 0:
                env.render()
            elif opt.render_mode == 1:
                img.set_data(env.render(mode='rgb_array'))
                display.display(plt.gcf())
                display.clear_output(wait=True)
            for agent_id, dqn in enumerate(dqn_agent_list):
                dqn.push_memory(s[agent_id], a_list[agent_id][0], r[agent_id], s_[agent_id])
            s = s_
    logger.info("经验池填充完成，开始训练！")
    setbar = tqdm(range(opt.episodes))
    cumulative_headways = [[] for _ in range(opt.num_agent)]
    cumulative_data = {
        'each_avg_reward': [[] for _ in range(opt.num_agent)],
        'each_total_reward': [[] for _ in range(opt.num_agent)],
    }
    cumulative_speeds = [[] for _ in range(opt.num_agent)]
    for ep_num in setbar:
        setbar.set_description(f"Episode: {ep_num + 1}", refresh=True)
        average_headways = [[] for _ in range(opt.num_agent)]
        s = env.reset()[0]
        total_reward = {}
        _speeds = []
        for i in range(opt.num_agent):
            total_reward[i] = []
            _speeds.append([])
            average_headways[i] = []
        total_headways = [0 for _ in range(opt.num_agent)]
        step_counts = [0 for _ in range(opt.num_agent)]
        done = False
        truncated = False
        step_num = 0
        while not done and not truncated:
            e = min(opt.max_exploration, np.exp(-count / opt.exploration_decay) + opt.min_exploration)
            a_list = []
            for agent_id, dqn in enumerate(dqn_agent_list):
                a, is_random = dqn.choose_action(s[agent_id], e)
                a_list.append((a, is_random))
            s_, r, done, truncated, info = env.step(tuple(map(lambda x: x[0], a_list)))
            if opt.render_mode == 0:
                env.render()
            elif opt.render_mode == 1:
                img.set_data(env.render(mode='rgb_array'))
                display.display(plt.gcf())
                display.clear_output(wait=True)
            for agent_id in range(opt.num_agent):
                headway = env._calculate_headway(env.controlled_vehicles[agent_id])
                if headway < float('inf'):
                    total_headways[agent_id] += headway
                    step_counts[agent_id] += 1
            postfix = ""
            step_num += 1
            for agent_id, dqn in enumerate(dqn_agent_list):
                dqn.push_memory(s[agent_id], a_list[agent_id][0], r[agent_id], s_[agent_id])
                total_reward[agent_id].append(r[agent_id])
                _speeds[agent_id].append(info["speed"][agent_id])
                count += 1
                loss_ = dqn.learn()
                postfix += f" {agent_id}:" + "{ " + f"a={action2action[a_list[agent_id][0]]}, is_random={a_list[agent_id][1]}, r={r[agent_id]}" + " }"
            setbar.set_postfix(step_num=step_num, e=e, info=postfix)
            s = s_
        for agent_id in range(opt.num_agent):
            if step_counts[agent_id] > 0:
                average_headway = total_headways[agent_id] / step_counts[agent_id]
                average_headways[agent_id].append(average_headway)
                cumulative_headways[agent_id].append(average_headway)
            else:
                average_headways[agent_id].append(None)
                cumulative_headways[agent_id].append(None)
        if done:
            crashed_num+=1
        for agent_id in range(opt.num_agent):
            each_avg_reward[agent_id].append(np.mean(total_reward[agent_id]))
            each_total_reward[agent_id].append(np.sum(total_reward[agent_id]))
            avg_speeds[agent_id].append(np.mean(_speeds[agent_id]))

        crasheds.append(crashed_num)
        steps.append(step_num)
        ep_list.append(ep_num)
        avg_avg_rewards.append(np.mean([each_avg_reward[agent_id][-1] for agent_id in range(opt.num_agent)]))
        total_avg_rewards.append(np.sum([each_avg_reward[agent_id][-1] for agent_id in range(opt.num_agent)]))
        avg_total_rewards.append(np.mean([each_total_reward[agent_id][-1] for agent_id in range(opt.num_agent)]))
        total_total_rewards.append(np.sum([each_total_reward[agent_id][-1] for agent_id in range(opt.num_agent)]))

        # 绘图任务触发，原本位于 if 条件内部
        for agent_id, dqn in enumerate(dqn_agent_list):
            plot_thread.push_task("each_avg_reward", "episodes", "avg_reward", ep_list, each_avg_reward[agent_id],
                                  os.path.join(save_dir, f"each_avg_reward_{agent_id}.png"))
            plot_thread.push_task("each_total_reward", "episodes", "total_reward", ep_list, each_total_reward[agent_id],
                                  os.path.join(save_dir, f"each_total_reward_{agent_id}.png"))
            plot_thread.push_task("each_avg_speed", "episodes", "avg_speed", ep_list, avg_speeds[agent_id],
                                  os.path.join(save_dir, f"each_avg_speed_{agent_id}.png"))

        plot_thread.push_task("crashed", "episodes", "crashed", ep_list, crasheds,
                              os.path.join(save_dir, f"crashed.png"))
        plot_thread.push_task("steps", "episodes", "steps", ep_list, steps, os.path.join(save_dir, f"steps.png"))
        plot_thread.push_task("avg_avg_rewards", "episodes", "avg_avg_reward", ep_list, avg_avg_rewards,
                              os.path.join(save_dir, f"avg_avg_rewards.png"))
        plot_thread.push_task("total_avg_rewards", "episodes", "total_avg_reward", ep_list, total_avg_rewards,
                              os.path.join(save_dir, f"total_avg_rewards.png"))
        plot_thread.push_task("avg_total_rewards", "episodes", "avg_total_reward", ep_list, avg_total_rewards,
                              os.path.join(save_dir, f"avg_total_rewards.png"))
        plot_thread.push_task("total_total_rewards", "episodes", "total_total_reward", ep_list, total_total_rewards,
                              os.path.join(save_dir, f"total_total_rewards.png"))
        for agent_id in range(opt.num_agent):
            cumulative_data['each_avg_reward'][agent_id].append(np.mean(total_reward[agent_id]))
            cumulative_data['each_total_reward'][agent_id].append(np.sum(total_reward[agent_id]))
            cumulative_speeds[agent_id].append(np.mean(_speeds[agent_id]))
        if (ep_num + 1) % 10000 == 0:
            training_data = {
                'each_avg_reward': [cumulative_data['each_avg_reward'][agent_id][:ep_num + 1] for agent_id in range(opt.num_agent)],
                'each_total_reward': [cumulative_data['each_total_reward'][agent_id][:ep_num + 1] for agent_id in range(opt.num_agent)],
                'each_avg_speed': [cumulative_speeds[agent_id][:ep_num + 1] for agent_id in range(opt.num_agent)],
            }
            for agent_id in range(opt.num_agent):
                excel_filename = os.path.join(save_dir, f'agent_{agent_id}_up_to_episode_{ep_num + 1}.xlsx')
                agent_data = {
                    'avg_reward': training_data['each_avg_reward'][agent_id],
                    'total_reward': training_data['each_total_reward'][agent_id],
                    'avg_speed': training_data['each_avg_speed'][agent_id],
                    'avg_headway': cumulative_headways[agent_id]
                }
                if all(len(lst) == len(agent_data['avg_reward']) for lst in agent_data.values()):
                    save_to_excel(agent_data, excel_filename)
    logger.info("训练结束！")
def test(opt):
    save_dir = get_log_path(os.path.join(opt.save_path, "test"))
    model_dir = os.path.join(opt.test_model_dir, MODELS_DIR)
    opt_dict, multi_highway_config = load_config(model_dir)
    opt.input_size = opt_dict["input_size"]
    opt.output_size = opt_dict["output_size"]
    opt.save_dir = save_dir
    logger = logger_setting(save_dir)
    print_args(logger, opt, multi_highway_config)
    env = Multi_Highway_Env(multi_highway_config)
    dqn_agent_list = [DQN(opt, str(i)) for i in range(opt.num_agent)]
    for dqn in dqn_agent_list:
        dqn.load_model(model_dir)
    e = 0
    setbar = tqdm(range(opt.episodes))
    for ep_num in setbar:
        setbar.set_description(f"Episode: {ep_num + 1}", refresh=True)
        s = env.reset()[0]
        done = False
        truncated = False
        step_num = 0
        while not done and not truncated:
            a_list = []
            for agent_id, dqn in enumerate(dqn_agent_list):
                a, is_random = dqn.choose_action(s[agent_id], e)
                a_list.append((a, is_random))
            s_, r, done, truncated, info = env.step(tuple(map(lambda x: x[0], a_list)))
            if opt.render_mode == 0:
                env.render()
            elif opt.render_mode == 1:
                plt.figure(3)
                plt.clf()
                plt.imshow(env.render(mode='rgb_array'))
                plt.title("Episode: %d" % (ep_num))
                plt.axis('off')
                display.clear_output(wait=True)
                display.display(plt.gcf())
            postfix = ""
            step_num += 1
            for agent_id, dqn in enumerate(dqn_agent_list):
                postfix += f" {agent_id}:" + "{ " + f"a={action2action[a_list[agent_id][0]]}, r={r[agent_id]}" + " }"
            setbar.set_postfix(step_num=step_num, info=postfix)