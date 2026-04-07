import argparse
import json
from es_env_cmabNR import EnergySavingEnv
# Importiamo la nuova classe
from cmabNR import MultiAgentGameMAB 
import wandb
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/openlab/ns-o-ran-gym/src/environments/scenario_configurations/es_use_case.json")
    parser.add_argument("--output_folder", type=str, default="output")
    parser.add_argument("--ns3_path", type=str, default="/home/openlab/ns-3-mmwave-oran")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--optimized", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    try:
        with open(args.config) as params_file:
            params = params_file.read()
    except FileNotFoundError:
        exit(-1)
        
    base_scenario_configuration = json.loads(params)

    # --- INIZIO PIPELINE SEQUENZIALE ---
    lista_droni = [3, 5, 7, 9]
    
    for num_drones in lista_droni:
        print(f"\n{'='*60}")
        print(f" AVVIO TRAINING CON {num_drones} DRONI ")
        print(f"{'='*60}\n")
        
        # Iniettiamo il numero di droni corrente nella configurazione
        scenario_configuration = base_scenario_configuration.copy()
        scenario_configuration['nMmWaveEnbNodes'] = [num_drones]
        current_out_folder = f"{args.output_folder}_{num_drones}drones"
        print('Creating ES Environment')
        env = EnergySavingEnv(
            ns3_path=args.ns3_path,
            scenario_configuration=scenario_configuration,
            output_folder=current_out_folder,
            optimized=args.optimized,
            do_heuristic=True
        )

        obs, info = env.reset()

        print(f"Inizializzazione Multi-Agent Game MAB con {num_drones} agenti...")
        agent = MultiAgentGameMAB(num_agents=num_drones, epsilon_start=1.0, epsilon_end=0.05)

        wandb.init(
            name=f"GT-FANET_UE-{scenario_configuration['ues']}_{num_drones}-DRONES",
            project="energy-saving",
            tags=["game-theory", "multi-agent-mab", f"ue {scenario_configuration['ues']}"],
            entity="sakhann9698-universit-catania",
            reinit=True, # <--- FONDAMENTALE per riavviare un nuovo run nello stesso script
        )

        episode = 1
        negotiation_costs = np.zeros(agent.num_agents)
        
        # Esegue i 100 episodi per QUESTO set di droni
        while episode <= 100:
            cumulative_reward = 0.0
            active_rus_steps = []
            convergence_list = []
            action = np.ones(agent.num_agents, dtype=int) 
            
            for step in range(2, 99):
                print(f'\n[Droni: {num_drones}] Ep {episode}, Step {step} ', end='', flush=True)
                prev_obs = obs

                action, converged = agent.negotiate(previous_actions=action, cost_vector=negotiation_costs)
                
                obs, profit, terminated, truncated, info = env.step(action)
                if 'agent_costs' in info:
                    negotiation_costs = info['agent_costs']
                else:
                    negotiation_costs = np.zeros(agent.num_agents)

                global_reward = list(profit.values())[0][0] if profit else 0.0
                sum_individual_penalties = sum([val[1] for val in profit.values()])
                step_total_reward = global_reward + sum_individual_penalties
                cumulative_reward += float(step_total_reward)
                
                agent.update(profit, env.cellList)

                current_active_rus = sum(action)
                active_rus_steps.append(current_active_rus)
                convergence_list.append(1 if converged else 0)

                if terminated or truncated or env.crashed:
                    break

            # --- RACCOLTA E LOG DELLE METRICHE (Mantieni il tuo codice originale qui) ---
            avg_qos = float(np.mean(env.avg_qos)) if env.avg_qos else 0.0
            avg_nrbdl = float(np.mean(env.avg_nrbdl)) if env.avg_nrbdl else 0.0
            avg_rlf = float(np.mean(env.avg_rlf)) if env.avg_rlf else 0.0
            avg_energy_saving = float(np.mean(env.average_energy_consumption)) if env.average_energy_consumption else 0.0
            avg_active_rus = float(np.mean(active_rus_steps)) if active_rus_steps else 0.0
            convergence_rate = float(np.mean(convergence_list))
            
            log_dict = {
                "energy saving": avg_energy_saving,
                "episode": episode,
                "average QoS": avg_qos,
                "cumulative reward": float(cumulative_reward),
                "avg active RUs": avg_active_rus,
                "average RLF": avg_rlf,
                "average NRDBL": avg_nrbdl,
                "convergence rate": convergence_rate,
                "epsilon": float(agent.current_epsilon()),
            }

            
            if env.energy_consumption_per_cell:
                per_drone = np.array(env.energy_consumption_per_cell)
                mean_per_drone = np.mean(per_drone, axis=0)
                for i, cell_id in enumerate(env.cellList):
                    log_dict[f"average_power_consumption/drone_{cell_id}"] = float(mean_per_drone[i])

            if episode == 100:
                # 1. Format the array into a 2D list of [step_index, value]
                qos_data = [[i, val] for i, val in enumerate(env.avg_qos)]
                actCost_data = [[i, val] for i, val in enumerate(env.average_energy_consumption)]
                ru_data = [[i, val] for i, val in enumerate(active_rus_steps)]
                rlf_data = [[i, val] for i, val in enumerate(env.avg_rlf)]
                nrdbl_data = [[i, val] for i, val in enumerate(env.avg_nrbdl)]

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
                log_dict["NRDBL Distribution"] = wandb.Histogram(env.avg_nrbdl)

            if env.crashed == False and step >= 80:
                wandb.log(log_dict, commit=True)
                episode += 1
            else:
                print(f"[WARNING] Crash ep {episode}")

            env._reset_stats()
            obs, info = env.reset()
            
        # FINE DEI 100 EPISODI PER QUESTI DRONI -> Chiudiamo il run su WandB
        print(f"\nCompletato addestramento con {num_drones} droni.")
        wandb.finish()