import argparse
import json
from gymnasium.utils.env_checker import check_env
from es_env_dqn import EnergySavingEnv
from discrete_dqn import DQNAgent
from ppo_discrete import PPOAgent
import wandb
import statistics
import numpy as np

def to_float_flat(obs):
    arr = np.array(obs, dtype=np.float32)
    return arr.flatten()

if __name__ == '__main__':
    #######################
    # Parse arguments #
    #######################
    parser = argparse.ArgumentParser(description="Run the traffic steering environment")
    parser.add_argument("--config", type=str, default="/home/openlab/ns-o-ran-gym/src/environments/scenario_configurations/es_use_case.json",
                        help="Path to the configuration file")
    parser.add_argument("--output_folder", type=str, default="output",
                        help="Path to the output folder")
    parser.add_argument("--ns3_path", type=str, default="/home/openlab/ns-3-mmwave-oran",
                        help="Path to the ns-3 mmWave O-RAN environment")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of steps to run in the environment")
    parser.add_argument("--optimized", action="store_true",
                        help="Enable optimization mode")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--alg", type=str, choices=["PPO", "DQN"],
                        help="Algorithm type (PPO, DQN)")
    
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    
    args = parser.parse_args()

    configuration_path = args.config
    output_folder = args.output_folder
    ns3_path = args.ns3_path
    num_steps = args.num_steps # maximum number of steps for each environment is calculated using the indication periodicity, we refer here to training steps
    optimized = args.optimized
    verbose = args.verbose
    alg = args.alg

    try:
        with open(configuration_path) as params_file:
            params = params_file.read()
    except FileNotFoundError:
        print(f"Cannot open '{configuration_path}' file, exiting")
        exit(-1)

    base_scenario_configuration = json.loads(params)
    lista_droni = [3, 5, 7, 9]
for num_drones in lista_droni:
        print(f"\n{'='*60}")
        print(f" AVVIO TRAINING {args.alg} CON {num_drones} DRONI ")
        print(f"{'='*60}\n")

        scenario_configuration = base_scenario_configuration.copy()
        scenario_configuration['nMmWaveEnbNodes'] = [num_drones]

        current_out_folder = f"{output_folder}_{num_drones}drones"

        print('Creating ES Environment')
        env = EnergySavingEnv(ns3_path=ns3_path, scenario_configuration=scenario_configuration,
                              output_folder=current_out_folder, optimized=optimized)

        print('Environment Created!')
        print('Launch reset ', end='', flush=True)
        obs, info = env.reset()
        print('done')

        state_dim = env.observation_space.shape[0]
        action_list = env.action_list
        n_actions = len(env.action_list)
        print("state space:", state_dim)
        print("action space:", n_actions)
        
        if args.alg == "PPO":
            agent = PPOAgent(
                state_dim=state_dim,
                n_actions=n_actions,
                learning_rate=1e-5,   
                batch_size=64,        
                gamma=0.99,           
                gae_lambda=0.95,      
                clip_range=args.clip_range,
                n_epochs=args.n_epochs,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
            )
            wandb_config = {
                "algo": "PPO",
                "lr": 1e-5,
                "batch_size": 64,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": args.clip_range,
                "n_epochs": args.n_epochs,
                "ent_coef": args.ent_coef,
                "vf_coef": args.vf_coef,
            }
        else: # DQN
            agent = DQNAgent(state_dim=state_dim, action_list=action_list)
            wandb_config = {"algo": "DQN"}

        wandb.init(
            name=f"{args.alg.lower()}-fanet_ue-{scenario_configuration['ues']}_{num_drones}DRONES",
            project="energy-saving",
            tags=[args.alg.lower(), f"ue {scenario_configuration['ues']}"],
            entity="sakhann9698-universit-catania",
            reinit=True,
            config=wandb_config
        )
        
        episode = 1
        while episode <= 100:
            cumulative_reward = 0
            active_rus_steps = []
            for step in range(2, 99):
                print(f'\n[Droni: {num_drones}] Ep {episode}, Step {step} ', end='', flush=True)
                s = to_float_flat(obs)

                action_index = agent.select_action(s)
                obs_next, reward, terminated, truncated, info = env.step(action_index)

                cumulative_reward += float(reward)
                done = bool(terminated or truncated)
                obs_next_flat = to_float_flat(obs_next)

                if args.alg == "PPO":
                    agent.store(reward=float(reward), done=done, next_state=obs_next_flat)
                else: # DQN
                    agent.replay_buffer.push(s, action_index, reward, obs_next_flat, terminated)
                    agent.update()
                
                dec_action = env.action_list[action_index]
                bin_format = f'0{len(env.cellList)}b'
                binary_action = format(dec_action, bin_format)
                active_rus = binary_action.count('1')
                active_rus_steps.append(active_rus)
                obs = obs_next

                if done or env.crashed:
                    break
                
            if args.alg == "PPO" and not env.crashed:
                stats = agent.update()
                if stats:
                    wandb.log({
                        "ppo/policy_loss": stats.get("policy_loss", 0.0),
                        "ppo/value_loss": stats.get("value_loss", 0.0),
                        "ppo/entropy": stats.get("entropy", 0.0),
                        "ppo/clip_frac": stats.get("clip_frac", 0.0),
                    }, commit=False)

            avg_qos = np.mean(env.avg_qos) if env.avg_qos else 0.0
            avg_nrdbl = np.mean(env.avg_nrdbl) if env.avg_nrdbl else 0.0
            avg_rlf = np.mean(env.avg_rlf) if env.avg_rlf else 0.0
            avg_energy_saving = np.mean(env.average_energy_consumption) if env.average_energy_consumption else 0.0
            avg_active_rus = float(np.mean(active_rus_steps)) if active_rus_steps else 0.0
            
            log_dict = {
                "activation cost": avg_energy_saving,
                "episode": episode,
                "average throughput": avg_qos,
                "cumulative reward": float(cumulative_reward),
                "avg active RUs": avg_active_rus,
                "average RLF": avg_rlf,
                "average NRDBL": avg_nrdbl,
            }

            if env.energy_consumption_per_cell:
                per_drone = np.array(env.energy_consumption_per_cell)
                mean_per_drone = np.mean(per_drone, axis=0)
                for i, cell_id in enumerate(env.cellList):
                    log_dict[f"average_energy_consumption/drone_{cell_id}"] = float(mean_per_drone[i])

            if episode == 100 and not env.crashed and step >= 80:
                # 1. Format the array into a 2D list of [step_index, value]
                qos_data = [[i, val] for i, val in enumerate(env.avg_qos)]
                actCost_data = [[i, val] for i, val in enumerate(env.average_energy_consumption)]
                ru_data = [[i, val] for i, val in enumerate(active_rus_steps)]
                rlf_data = [[i, val] for i, val in enumerate(env.avg_rlf)]
                nrdbl_data = [[i, val] for i, val in enumerate(env.avg_nrdbl)]

                # 2. Create a wandb Table
                qos_table = wandb.Table(data=qos_data, columns=["intra_episode_step", "QoS"])
                actCost_table = wandb.Table(data=actCost_data, columns=["intra_episode_step", "Activation Cost"])
                ru_table = wandb.Table(data=ru_data, columns=["intra_episode_step", "Number of RU"])
                rlf_table = wandb.Table(data=rlf_data, columns=["intra_episode_step", "RLF"])
                nrdbl_table = wandb.Table(data=nrdbl_data, columns=["intra_episode_step", "NRDBL"])

                # 3. Log it as a line plot
                log_dict["QoS_Curve"] = wandb.plot.line(
                    qos_table, 
                    x="intra_episode_step", 
                    y="QoS", 
                    title=f"QoS over Episode {episode}"
                )
                log_dict["actCost_Curve"] = wandb.plot.line(
                    actCost_table, 
                    x="intra_episode_step", 
                    y="Activation Cost", 
                    title=f"Activation Cost over Episode {episode}"
                )
                log_dict["ru_Curve"] = wandb.plot.line(
                    ru_table, 
                    x="intra_episode_step", 
                    y="Number of RU", 
                    title=f"Number of RU over Episode {episode}"
                )
                log_dict["rlf_Curve"] = wandb.plot.line(
                    rlf_table, 
                    x="intra_episode_step", 
                    y="RLF", 
                    title=f"RLF over Episode {episode}"
                )
                log_dict["nrdbl_Curve"] = wandb.plot.line(
                    nrdbl_table, 
                    x="intra_episode_step", 
                    y="NRDBL", 
                    title=f"NRDBL over Episode {episode}"
                )

                log_dict["QoS Distribution"] = wandb.Histogram(env.avg_qos)
                log_dict["ACTCOST Distribution"] = wandb.Histogram(env.average_energy_consumption)
                log_dict["RU Distribution"] = wandb.Histogram(active_rus_steps)
                log_dict["RLF Distribution"] = wandb.Histogram(env.avg_rlf)
                log_dict["NRDBL Distribution"] = wandb.Histogram(env.avg_nrdbl)

            if not env.crashed and step >= 80:
                wandb.log(log_dict, commit=True)
                episode += 1
            else:
                print(f"\n[WARNING] Crash ep {episode}")
                
            env._reset_stats()
            obs, info = env.reset()
            
        print(f"\nCompletato addestramento DQN con {num_drones} droni.")
        wandb.finish()