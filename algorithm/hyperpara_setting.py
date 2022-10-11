
import numpy as np
import copy
from env_utils.env_utils import load_scenario
from env_utils.dict2class import dict2obj


Cooperative_Navigation_setting = {
    "algorithm": "madre",
    "config": {
        "train":
            {
                "max_episodes": 100000,
                "before_train_episode": 200,
                "per_episode_length": 25,
                "reward_scale": 1.0,
                "single_step_mode": False,
                "gamma": 0.95,
                "TD": 1,
                "batch_norm": False,
                "tau": 0.01,
                "hard_action_interval": 0.3,
                "epsilon": 0.2,
                "joint_store": True,
                "buffer_select_with_batch_size":True,
                "batch_size": 4096,
                "mini_batch_size": 1024,
                "buffer_clear_rate": 0.4,
                "buffer_select_rate": 0.5,
                "lr_decay_interval": 3000,
                "decay_rate": 0.95,
                "lr_min": 1e-5,
                "greedy_grow_interval": 1000,
                "greedy": 0.7,
                "grow_rate": 1.01,
                "greedy_max": 0.9,
                "delta": 1e-6,
                "use_wandb": True
            },
        "control":
            {
                "summary_record": True,
                "reward_predict_mode": "net",
                "print_info_every_n_step": 300,
                "update_every_n_ep": 4,
                "update_repeat_n_time": 1,
                "clear_buffer_every_n_ep": 1000,
                "plot_every_n_step": 3000000000,
                "render_threshold": 9999999999,
            },
        "evaluate":
            {
                "render": False,
                "eval_every_n_ep": 50,
                "eval_episode": 10,
            },
        "save":
            {
                "total_model_num": 20,
                "save_data_every_n_step": 2000000000,
            },
        "env":
            {
                "domain": "MPE",
                "scenario": "simple_spread",
                "adversary": True,
                "agent_num": 1,
                "action_dim": None,
                "all_action_dim": None,
                "state_dim": None,
                "all_state_dim": None,

            },
        "graph":
            {
                "atten_head_num": 8,
                "in_norm": False,
                "in_drop": False,
                "coef_drop": False,
                "residual": True,
                "Con1d_kernel_num": 8,
            },
        "critic":
            {
                "input_dim": 10,
                "output_dim": 10,
                "layer_num": 2,
                "lr": 1e-3,
            },
        "actor":
            {
                "input_dim": 10,
                "output_dim": 10,
                "layer_num": 2,
                "lr": 1e-3,
            },
        "reward":
            {
                "input_dim": 10,
                "output_dim": 3,
                "layer_num": 2,
                "lr": 1e-3,
                "sdrr": 0.001,
                "dist_reward_fit": True,
                "global_reward_prediction": False,
                "reward_aggregation_type": "l_smo",  # available type = ["l_ss+g_ss", "l_smo+g_mo", "l_mo+g_mo", "l_smo+g_ss", "l_smo"]
                "team_reward": False,
                "reward_uncertainty_type": "r_ac-dist",  # available_type = ["r_dete", "r_dist", "r_ac-dist"]
                "eval_with_reward_uncertainty": False
            },
        "feature":
            {
                "data":{"input_dim": 10},
                "feature_extractor":{
                    "input_dim": 8,
                    "output_dim": 8,
                    "gaussian_num": 5,
                    "lr":1e-3,},
                "vae":{
                    "encoder":{"input_dim":10, "output_dim":8,},
                    "decoder":{"input_dim":8, "output_dim":10,},
                    "lr":1e-3,
                }
            }
    }
}

simple_refrence_setting = {
    "algorithm": "madre",
    "config": {
        "train":
            {
                "max_episodes":100000,
                "before_train_episode": 200,
                "per_episode_length": 25,
                "reward_scale": 1.0,
                "single_step_mode": False,
                "gamma": 0.95,
                "TD": 1,
                "batch_norm": False,
                "tau": 0.01,
                "hard_action_interval": 0.3,
                "epsilon": 0.2,
                "joint_store": True,
                "buffer_select_with_batch_size":True,
                "batch_size": 4096,
                "mini_batch_size": 1024,
                "buffer_clear_rate": 0.4,
                "buffer_select_rate": 0.5,
                "lr_decay_interval": 3000,
                "decay_rate": 0.95,
                "lr_min": 1e-5,
                "greedy_grow_interval": 1000,
                "greedy": 0.7,
                "grow_rate": 1.01,
                "greedy_max": 0.9,
                "delta": 1e-6,
                "use_wandb": True
            },
        "control":
            {
                "summary_record": True,
                "reward_predict_mode": "net",
                "print_info_every_n_step": 300,
                "update_every_n_ep": 4,
                "update_repeat_n_time": 1,
                "clear_buffer_every_n_ep": 1000,
                "plot_every_n_step": 3000000000,
                "render_threshold": 99999999999,
            },
        "evaluate":
            {
                "render": False,
                "eval_every_n_ep": 50,
                "eval_episode": 5,
            },
        "save":
            {
                "total_model_num": 20,
                "save_data_every_n_step": 2000000000,
            },
        "env":
            {
                "domain": "MPE",
                "scenario": "simple_reference",
                "adversary": True,
                "agent_num": 1,
                "action_dim": None,
                "all_action_dim": None,
                "state_dim": None,
                "all_state_dim": None,

            },
        "graph":
            {
                "atten_head_num": 8,
                "in_norm": False,
                "in_drop": False,
                "coef_drop": False,
                "residual": True,
                "Con1d_kernel_num": 8,
            },
        "critic":
            {
                "input_dim": 10,
                "output_dim": 10,
                "layer_num": 2,
                "lr": 1e-3,
            },
        "actor":
            {
                "input_dim": 10,
                "output_dim": 10,
                "layer_num": 2,
                "lr": 1e-3,
            },
        "reward":
            {
                "input_dim": 10,
                "output_dim": 3,
                "layer_num": 2,
                "lr": 1e-3,
                "sdrr": 0.001,
                "dist_reward_fit": True,
                "global_reward_prediction": False,
                "reward_aggregation_type": "l_smo",  # available type = ["l_ss+g_ss", "l_smo+g_mo", "l_mo+g_mo", "l_smo+g_ss", "l_smo"]
                "team_reward": False,
                "reward_uncertainty_type": "r_ac-dist",  # available_type = ["r_dete", "r_dist", "r_ac-dist"]
                "eval_with_reward_uncertainty": False
            }
    }
}

treasure_setting = {
    "algorithm": "madae",
    "config": {
        "train":
            {
                "max_episodes":100000,
                "before_train_episode": 200,
                "per_episode_length": 25,
                "reward_scale": 1.0,
                "single_step_mode": False,
                "gamma": 0.95,
                "TD": 1,
                "batch_norm": False,
                "tau": 0.01,
                "hard_action_interval": 0.3,
                "epsilon": 0.2,
                "joint_store": True,
                "buffer_select_with_batch_size":True,
                "batch_size": 4096,
                "mini_batch_size": 1024,
                "buffer_clear_rate": 0.4,
                "buffer_select_rate": 0.5,
                "lr_decay_interval": 3000,
                "decay_rate": 0.95,
                "lr_min": 1e-5,
                "greedy_grow_interval": 1000,
                "greedy": 0.7,
                "grow_rate": 1.01,
                "greedy_max": 0.9,
                "delta": 1e-6,
                "use_wandb": True
            },
        "control":
            {
                "summary_record": True,
                "reward_predict_mode": "net",
                "print_info_every_n_step": 300,
                "update_every_n_ep": 4,
                "update_repeat_n_time": 1,
                "clear_buffer_every_n_ep": 1000,
                "plot_every_n_step": 30000000000,
                "render_threshold": 99999999999,
            },
        "evaluate":
            {
                "render": False,
                "eval_every_n_ep": 50,
                "eval_episode": 10,
            },
        "save":
            {
                "total_model_num": 20,
                "save_data_every_n_step": 2000000000,
            },
        "env":
            {
                "domain": "MPE",
                "scenario": "fullobs_collect_treasure",
                "adversary": False,
                "agent_num": 1,
                "action_dim": None,
                "all_action_dim": None,
                "state_dim": None,
                "all_state_dim": None,

            },
        "graph":
            {
                "atten_head_num": 8,
                "in_norm": False,
                "in_drop": False,
                "coef_drop": False,
                "residual": True,
                "Con1d_kernel_num": 8,
            },
        "critic":
            {
                "input_dim": 10,
                "output_dim": 10,
                "layer_num": 2,
                "lr": 1e-3,
            },
        "actor":
            {
                "input_dim": 10,
                "output_dim": 10,
                "layer_num": 2,
                "lr": 1e-3,
            },
        "reward":
            {
                "input_dim": 10,
                "output_dim": 3,
                "layer_num": 2,
                "lr": 3e-4,  #1e-3
                "sdrr": 0.001,
                "dist_reward_fit": False,
                "global_reward_prediction": True,
                "reward_aggregation_type": "l_smo+g_mo",  # available type = ["l_ss+g_ss", "l_smo+g_mo", "l_mo+g_mo", "l_smo+g_ss", "l_smo"]
                "team_reward": False,
                "reward_uncertainty_type": "r_ac-dist",  # available_type = ["r_dete", "r_dist", "r_ac-dist"]
                "eval_with_reward_uncertainty": False
            },
        "feature":
            {
                "data":{"input_dim": 10},
                "feature_extractor":{
                    "input_dim": 8,
                    "output_dim": 8,
                    "gaussian_num": 5,
                    "lr":1e-3,},
                "vae":{
                    "encoder":{"input_dim":10, "output_dim":8,},
                    "decoder":{"input_dim":8, "output_dim":10,},
                    "lr":1e-3,
                }
            }
    }
}

def load_MPE_scenario(setting_info):
    env = load_scenario(env_name=setting_info["config"]["env"]["scenario"],
                        env_category=setting_info["config"]["env"]["domain"])
    if setting_info["config"]["env"]["scenario"] == "simple_spread" or setting_info["config"]["env"]["scenario"] == "fullobs_collect_treasure":
        state_dim = env.observation_space[0].shape[0]
        action_dim = env.action_space[0].n
        all_state_dim = [env.observation_space[i].shape[0] for i in range(env.n)]
        all_action_dim = [env.action_space[i].n for i in range(env.n)]
        setting_info["config"]["env"].update({
            "agent_num": env.n, "state_dim": state_dim, "all_state_dim": all_state_dim, "action_dim": action_dim,
            "all_action_dim": all_action_dim, "action_type": "discrete", "n_action": 1})
        setting_info["config"]["critic"].update({"input_dim": state_dim, "output_dim": 1})
        setting_info["config"]["actor"].update({"input_dim": state_dim, "output_dim": action_dim})
        setting_info["config"]["reward"].update({"input_dim": state_dim, "output_dim": 3})
        setting_info["config"]["feature"].update({"data": {"input_dim": state_dim+1}, "feature_extractor": {"input_dim": state_dim+1, "output_dim": state_dim+1, "gaussian_num": 5, "lr":1e-3},
                                                  "vae":{"lr":1e-3, "encoder": {"input_dim": state_dim+1, "output_dim": state_dim+1}, "decoder": {"input_dim": state_dim+1, "output_dim": state_dim+1}}})
    elif setting_info["config"]["env"]["scenario"] == "simple_reference":
        state_dim = env.observation_space[0].shape[0]
        action_ncat = [int(_) for _ in (env.action_space[0].high - env.action_space[0].low + 1)]
        action_dim = int(np.sum([_ for _ in action_ncat]))
        all_state_dim = [env.observation_space[i].shape[0] for i in range(env.n)]
        all_action_dim = [int(np.sum(act_space.high - act_space.low + 1)) for act_space in env.action_space]
        setting_info["config"]["env"].update({
            "agent_num": env.n, "state_dim": state_dim, "all_state_dim": all_state_dim, "action_dim": action_dim,
            "all_action_dim": all_action_dim, "action_type":"multi_discrete", "n_action": len(action_ncat), "action_ncat": action_ncat})
        setting_info["config"]["critic"].update({"input_dim": state_dim, "output_dim": 1})
        setting_info["config"]["actor"].update({"input_dim": state_dim, "output_dim": action_dim})
        setting_info["config"]["reward"].update({"input_dim": state_dim, "output_dim": 3})
    else:
        if setting_info["config"]["env"]["adversary"]:
            state_dim = env.observation_space[0].shape[0] if env.agents[0].adversary else \
            env.observation_space[-1].shape[0]
            action_dim = env.action_space[0].n if env.agents[0].adversary else env.action_space[-1].n
            agent_num = np.sum([1 if agent.adversary else 0 for agent in env.agents])
            all_state_dim, all_action_dim = [], []
            for i, agent in enumerate(env.agents):
                if agent.adversary:
                    all_state_dim.append(env.observation_space[i].shape[0])
                    all_action_dim.append(env.action_space[i].n)
        else:
            state_dim = env.observation_space[0].shape[0] if not env.agents[0].adversary else \
            env.observation_space[-1].shape[0]
            action_dim = env.action_space[0].n if not env.agents[0].adversary else env.action_space[-1].n
            agent_num = np.sum([0 if agent.adversary else 1 for agent in env.agents])
            all_state_dim, all_action_dim = [], []
            for i, agent in enumerate(env.agents):
                if not agent.adversary:
                    all_state_dim.append(env.observation_space[i].shape[0])
                    all_action_dim.append(env.action_space[i].n)
        new_env_info = {"agent_num": agent_num,
                        "state_dim": state_dim,
                        "all_state_dim": all_state_dim,
                        "action_dim": action_dim,
                        "all_action_dim": all_action_dim,
                        }
        setting_info["config"]["env"].update(new_env_info)
        setting_info["config"]["critic"].update({"input_dim": state_dim, "output_dim": 1})
        setting_info["config"]["actor"].update({"input_dim": state_dim, "output_dim": action_dim})
        setting_info["config"]["reward"].update({"input_dim": state_dim, "output_dim": 3})

    return setting_info

def get_variant_config(scenario="CoopNavi"):
    if scenario == "CoopNavi":
        variant_config = load_MPE_scenario(setting_info=Cooperative_Navigation_setting)
    elif scenario == "reference":
        variant_config = load_MPE_scenario(setting_info=simple_refrence_setting)
    elif scenario == "fullobs_collect_treasure":
        variant_config = load_MPE_scenario(setting_info=treasure_setting)
    else:
        raise Exception("Please input right scenario name!")
    variant_config["config"].update({"config_record": copy.deepcopy(variant_config)})
    variant_obj = dict2obj(variant_config)
    return variant_obj.algorithm, variant_obj.config








