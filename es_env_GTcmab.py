from typing_extensions import override
import numpy as np
import pandas as pd
from nsoran.ns_env import NsOranEnv 
import glob
import csv
import os
import warnings
from gymnasium import spaces
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
class EnergySavingEnv(NsOranEnv):
    
    gnb_state_keys = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER",
            "cellId": "INTEGER",
            "state": "INTEGER",
            "energyFraction": "REAL",
            "remainingEnergy": "REAL"
        } 
        
    def __init__(self, ns3_path:str, scenario_configuration:dict, output_folder:str, optimized:bool, do_heuristic:bool = False):
        super().__init__(ns3_path=ns3_path, scenario='scenario-fanet', scenario_configuration=scenario_configuration,
                output_folder=output_folder, optimized=optimized,
                control_header = ['timestamp','cellId','hoAllowed'], log_file='EsActions.txt', control_file='es_actions_for_ns3.csv')
        
        self.folder_name = "Simulation"
        self.ns3_simulation_time = scenario_configuration['simTime']*1000
        num_drones = int(self.scenario_configuration.get('nMmWaveEnbNodes', 9))

        self.cellList = [i + 2 for i in range(num_drones)]
        base_columns = [
            'EEKPI_RL', 'ES_ON_COST', 'QosFlow.PdcpPduVolumeDL_Filter',
            'RLF_Counter', 'RLF_VALUE', 'RRU_PRBTOTDL', 'RRU.PrbUsedDl',
            'TB_TOTNBRDLINITIAL_64QAM_RATIO'
        ]

        self.columns_state = []
        for base_col in base_columns:
            for cell in self.cellList:
                self.columns_state.append(f'{base_col}_{cell}') #crea lo stato come sotto ma dinamicamente in base a scenario_configuration.get('nMmWaveEnbNodes', 9)
        self.columns_state.extend([
            'SUM_QosFlow.PdcpPduVolumeDL_Filter',
            'SUM_RLF_VALUE',
            'SUM_TB.TotNbrDl.1',
            'SUM_ES_ON_COST',
            'ZERO_COUNT'
        ])

        '''
        self.columns_state = [
            'EEKPI_RL_2', 'EEKPI_RL_3', 'EEKPI_RL_4', 'EEKPI_RL_5', 'EEKPI_RL_6', 'EEKPI_RL_7', 'EEKPI_RL_8', 'EEKPI_RL_9', 'EEKPI_RL_10',
            'ES_ON_COST_2', 'ES_ON_COST_3', 'ES_ON_COST_4', 'ES_ON_COST_5', 'ES_ON_COST_6', 'ES_ON_COST_7', 'ES_ON_COST_8', 'ES_ON_COST_9', 'ES_ON_COST_10',

            'QosFlow.PdcpPduVolumeDL_Filter_2', 'QosFlow.PdcpPduVolumeDL_Filter_3', 'QosFlow.PdcpPduVolumeDL_Filter_4', 'QosFlow.PdcpPduVolumeDL_Filter_5', 
            'QosFlow.PdcpPduVolumeDL_Filter_6', 'QosFlow.PdcpPduVolumeDL_Filter_7', 'QosFlow.PdcpPduVolumeDL_Filter_8', 'QosFlow.PdcpPduVolumeDL_Filter_9', 'QosFlow.PdcpPduVolumeDL_Filter_10',

            'RLF_Counter_2', 'RLF_Counter_3', 'RLF_Counter_4', 'RLF_Counter_5', 'RLF_Counter_6', 'RLF_Counter_7', 'RLF_Counter_8', 'RLF_Counter_9', 'RLF_Counter_10',

            'RLF_VALUE_2', 'RLF_VALUE_3', 'RLF_VALUE_4', 'RLF_VALUE_5', 'RLF_VALUE_6', 'RLF_VALUE_7', 'RLF_VALUE_8', 'RLF_VALUE_9', 'RLF_VALUE_10',

            'RRU_PRBTOTDL_2', 'RRU_PRBTOTDL_3', 'RRU_PRBTOTDL_4', 'RRU_PRBTOTDL_5', 'RRU_PRBTOTDL_6', 'RRU_PRBTOTDL_7', 'RRU_PRBTOTDL_8', 'RRU_PRBTOTDL_9', 'RRU_PRBTOTDL_10',

            'RRU.PrbUsedDl_2', 'RRU.PrbUsedDl_3', 'RRU.PrbUsedDl_4', 'RRU.PrbUsedDl_5', 'RRU.PrbUsedDl_6', 'RRU.PrbUsedDl_7', 'RRU.PrbUsedDl_8', 'RRU.PrbUsedDl_9', 'RRU.PrbUsedDl_10',

            'TB_TOTNBRDLINITIAL_64QAM_RATIO_2', 'TB_TOTNBRDLINITIAL_64QAM_RATIO_3', 'TB_TOTNBRDLINITIAL_64QAM_RATIO_4', 'TB_TOTNBRDLINITIAL_64QAM_RATIO_5', 'TB_TOTNBRDLINITIAL_64QAM_RATIO_6', 
            'TB_TOTNBRDLINITIAL_64QAM_RATIO_7', 'TB_TOTNBRDLINITIAL_64QAM_RATIO_8', 'TB_TOTNBRDLINITIAL_64QAM_RATIO_9', 'TB_TOTNBRDLINITIAL_64QAM_RATIO_10',

            'SUM_QosFlow.PdcpPduVolumeDL_Filter',
            'SUM_RLF_VALUE',
            'SUM_TB.TotNbrDl.1',
            'SUM_ES_ON_COST',
            'ZERO_COUNT'
        ]'''

        self.columns_reward = ['SUM_QosFlow.PdcpPduVolumeDL_Filter',
                               'SUM_TB.TotNbrDl.1',
                               'SUM_RLF_VALUE',
                               'SUM_ES_ON_COST',
                               'ZERO_COUNT'
                               ]
        
        #self.action_list = [i for i in range(128)] 
        #self.cellList = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        # If you ever use the discrete action index, this matches 2^num_cells
        self.action_list = [i for i in range(2 ** len(self.cellList))]
        # Per-cell drone battery: remainingEnergy (J), energyFraction [0,1]
        self.columns_energy = [f'remainingEnergy_{c}' for c in self.cellList] + [f'energyFraction_{c}' for c in self.cellList]
        self.columns_state = self.columns_state + self.columns_energy
        self.observations = []
        self.cells_states = {}
        self.previous_energy_joules = None

        obs_size = len(self.columns_state)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_list))

        self.cell_timestamp_state_dict = {cell: float('inf') for cell in self.cellList} 
        self.Cf = 1
        self.lambdaf = 0.1
        self.time_factor = 0.01
        self.heur = do_heuristic
        self.num_steps = 0
        self.previous_inverted_action = "0" * len(self.cellList)
        self.debug_energy = os.getenv("ES_DEBUG_ENERGY", "0") == "1"

        self.avg_qos = []
        self.avg_nrbdl = []
        self.avg_rlf = []
        self.average_energy_consumption = []
        # per-step ES_ON_COST per drone, shape: (steps, len(cellList))
        self.energy_consumption_per_cell = []
        self.crashed = False
        self.latest_individual_costs = np.zeros(len(self.cellList), dtype=np.float32)

        self.current_action = []

    def _reset_stats(self):
        self.avg_qos = []
        self.avg_nrbdl = []
        self.avg_rlf = []
        self.average_energy_consumption = []
        self.energy_consumption_per_cell = []
        self.crashed = False
        self.previous_energy_joules = None
        self.current_action = []
        self.latest_ue_counts = {}

    @override
    def _compute_action(self, action):
        """
        Converte l'azione (lista/array di 0 e 1) nel formato per NS-3.
        MAB Logic: 1 = Active (ON), 0 = Energy Saving (OFF)
        NS3 Logic: 0 = Normal/Active, 1 = Energy Saving Mode
        """
        cell_act_comb_lst = []

        # Gestione input da CMAB (lista o numpy array)
        if isinstance(action, list) or isinstance(action, np.ndarray):
            # Safety: evita che tutti i droni siano OFF contemporaneamente
            if np.sum(action) == 0:
                if isinstance(action, np.ndarray):
                    action = action.copy()
                    action[0] = 1
                else:
                    action = list(action)
                    action[0] = 1

            for i, val in enumerate(action):
                cell_id = self.cellList[i]
                
                # Inversione Logica:
                # Se Agente dice 1 (ON) -> NS3 vuole 0 (Non è in saving)
                # Se Agente dice 0 (OFF) -> NS3 vuole 1 (È in saving)
                ns3_val = 0 if int(val) == 1 else 1
                cell_act_comb_lst.append([cell_id, ns3_val])
            
            # Aggiorniamo la stringa binaria per statistiche (ZERO_COUNT)
            # ZERO_COUNT nel codice originale contava gli '0'.
            # Manteniamo la coerenza: '1' se spento (val=0), '0' se acceso (val=1)
            # La stringa rappresenta lo stato di Energy Saving (1=Saving/Off)
            self.previous_inverted_action = "".join(["1" if int(x) == 0 else "0" for x in action])
            
        else:
            # Fallback per compatibilità (se action fosse un intero)
            dec_action = self.action_list[action]
            # Convertiamo in binario riempiendo con "0" dinamicamente in base a quanti droni abbiamo
            bin_format = f'0{len(self.cellList)}b'
            bin_actions = [int(i) for i in list(format(dec_action, bin_format))]

        self.current_action = cell_act_comb_lst
            
        return cell_act_comb_lst
    
    def _update_cell_states(self):
        cell_states_table = self.datalake.read_table('bsState')
        target_ts = self.last_timestamp - 100
        latest_states = {}
        for cell_state in cell_states_table:
            # Expected row layout from table "bsState":
            # (timestamp, ueImsiComplete, cellId, state, ...)
            if len(cell_state) >= 4 and cell_state[0] == target_ts:
                latest_states[int(cell_state[2])] = int(cell_state[3])

        if not self.cells_states:
            # Initialize defaults once; do not reset all cells on partial snapshots.
            self.cells_states = {cell_id: 1 for cell_id in self.cellList}

        for cell_id in self.cellList:
            if cell_id in latest_states:
                self.cells_states[cell_id] = latest_states[cell_id]

        if self.debug_energy and len(latest_states) != len(self.cellList):
            print(
                f"[ES-DEBUG] partial bsState at ts={target_ts}: "
                f"got={len(latest_states)}/{len(self.cellList)}"
            )

    @override
    def _get_obs(self):
        kpms_raw = ["nrCellId", "QosFlow.PdcpPduVolumeDL_Filter", "TB.TotNbrDl.1", "L3 serving SINR", "RRU.PrbUsedDl", "TB.TotNbrDlInitial.64Qam", "TB.TotNbrDlInitial.Qpsk", "TB.TotNbrDlInitial.16Qam"]  
        ue_kpms = self.datalake.read_kpms(self.last_timestamp, kpms_raw) 
        self._update_cell_states()  
                   
        ue_complete_kpms = []
        if ue_kpms is None or len(ue_kpms) == 0:
            if self.num_steps > 20: 
                print(f"No UE KPMs found at step {self.num_steps}. Marking crashed.")
                self.crashed = True
            
            self.latest_ue_counts = {}
            # Restituiamo un'osservazione di zeri per non interrompere il loop
            self.observations = pd.DataFrame(
                [np.zeros(len(self.columns_state), dtype=np.float32)],
                columns=self.columns_state,
            )
            return self.observations.iloc[0].values
        else:
            for ue_kpm in ue_kpms:
                state = self.cells_states.get(ue_kpm[1], ()) 
                new_ue_kpm = ue_kpm + (state,)
                ue_complete_kpms.append(new_ue_kpm)   
        
        columns = ['ueImsiComplete'] + kpms_raw + ['state']
        df = pd.DataFrame(ue_complete_kpms, columns=columns)
        df["timestamp"] = self.last_timestamp

        self.latest_ue_counts = df['nrCellId'].dropna().astype(int).value_counts().to_dict()
        df, columns= self.getRLFCounter(df, columns)
        df = self.ue_centric_tocell_centric(df)
        self.observations = self.offline_training_preprocessing(df)

        # Append per-cell battery from datalake "bsState" table (point 2 design).
        # Expected row layout:
        # (timestamp, ueImsiComplete, cellId, state, energyFraction, remainingEnergy)
        bs_rows = self.datalake.read_table('bsState')
        latest_by_cell = {}  # cellId -> (timestamp, remainingEnergy, energyFraction)
        for row in bs_rows:
            if len(row) < 6:
                # Old rows without energy fields; skip and keep defaults.
                continue
            ts = row[0]
            cell_id = row[2]
            fraction = row[4]
            remaining_j = row[5]
            if ts <= self.last_timestamp and cell_id in self.cellList:
                if cell_id not in latest_by_cell or ts > latest_by_cell[cell_id][0]:
                    latest_by_cell[cell_id] = (ts, remaining_j, fraction)
        for cell in self.cellList:
            if cell in latest_by_cell:
                _, rem, frac = latest_by_cell[cell]
                self.observations[f'remainingEnergy_{cell}'] = np.float32(rem)
                self.observations[f'energyFraction_{cell}'] = np.float32(frac)
            else:
                self.observations[f'remainingEnergy_{cell}'] = np.float32(0.0)
                self.observations[f'energyFraction_{cell}'] = np.float32(0.0)

        return self.observations.iloc[0].values
        
    @override
    def _compute_reward(self):
        self.num_steps += 1
        cell_df = self.observations[self.columns_reward].copy()
        if self.observations is None or self.observations.empty:
            # Ritorna reward neutre per evitare errori di indicizzazione iloc[0]
            return {cell: np.array([0.0, 0.0], dtype=np.float32) for cell in self.cellList}
        db_row = {}
        db_row['timestamp'] = self.last_timestamp
        db_row['ueImsiComplete'] = None
        db_row['time_grafana'] = self.last_timestamp
        db_row['step'] = self.num_steps
        db_row['throughput'] = float(cell_df['SUM_QosFlow.PdcpPduVolumeDL_Filter'].iloc[0]) * 10 / 10**6
        db_row['en_cons'] = float(cell_df['SUM_TB.TotNbrDl.1'].iloc[0])
        db_row['rlf'] = float(cell_df['SUM_RLF_VALUE'].iloc[0])
        db_row['on_cost'] = float(cell_df['SUM_ES_ON_COST'].iloc[0])
        
        reward_0_global = float(
            0.7 * cell_df['SUM_QosFlow.PdcpPduVolumeDL_Filter'].iloc[0]
            - 0.15 * cell_df['SUM_RLF_VALUE'].iloc[0]
            - 0.15 * cell_df['SUM_TB.TotNbrDl.1'].iloc[0]
        )

        energy_joules_list = []
        for cell in self.cellList:
            # Use remainingEnergy to get actual Joules instead of the 0-1 fraction
            col = f'remainingEnergy_{cell}' 
            if col in self.observations:
                energy_joules_list.append(float(self.observations[col].iloc[0]))
            else:
                # Sostituisci 1.0 con la CAPACITÀ MASSIMA IN JOULE della batteria!
                energy_joules_list.append(1.0) 

        energy_joules = np.array(energy_joules_list, dtype=np.float32)
        energy_joules = np.nan_to_num(energy_joules, nan=1.0, posinf=1.0, neginf=1.0) # Idem qui per i NaN

        # 3. CALCOLO DELLA POTENZA (WATT) INDIVIDUALE
        step_duration_seconds = 0.1
        
        if self.previous_energy_joules is None:
            self.previous_energy_joules = energy_joules

        consumed_energy_joules = self.previous_energy_joules - energy_joules
        consumed_energy_joules = np.clip(consumed_energy_joules, 0.0, None)
        
        self.previous_energy_joules = energy_joules

        # Calculate Watts (Joules consumed / step duration in seconds)
        consumed_power_watts = consumed_energy_joules / step_duration_seconds

        # 4. CREAZIONE DELLE REWARD PER SINGOLO DRONE
        # Dizionario: { 'drone_1': [reward_0, reward_1_drone1], 'drone_2': [reward_0, reward_1_drone2]... }
        rewards_per_drone = {}
        
        for idx, cell in enumerate(self.cellList):
            # La penalità è basata ESCLUSIVAMENTE sul consumo di questo specifico drone
            # Il peso (es. -0.001) va calibrato in base all'ordine di grandezza dei Watt
            reward_1_individual = float(-0.001 * consumed_energy_joules[idx])
            
            # Vettore di reward per questo drone (posizione 0: globale, posizione 1: individuale)
            rewards_per_drone[cell] = np.array([reward_0_global, reward_1_individual], dtype=np.float32)

        individual_costs = []
        # Use self.observations for per-cell columns (cell_df only has columns_reward)
        obs = self.observations

        for cell in self.cellList:
            #Switching Cost
            es_cost_col = f'ES_ON_COST_{cell}'
            es_cost_val = float(obs[es_cost_col].iloc[0]) if es_cost_col in obs else 0.0
            
            agent_cost = 0.1 * es_cost_val
            individual_costs.append(agent_cost)

        self.latest_individual_costs = np.array(individual_costs, dtype=np.float32)
        self.datalake.insert_data("grafana", db_row)

        self.avg_qos.append(cell_df['SUM_QosFlow.PdcpPduVolumeDL_Filter'].iloc[0])
        self.avg_nrbdl.append(cell_df['SUM_TB.TotNbrDl.1'].iloc[0])
        self.avg_rlf.append(cell_df['SUM_RLF_VALUE'].iloc[0])
        self.average_energy_consumption.append(cell_df['SUM_ES_ON_COST'].iloc[0])
        
        # Per-drone consumed power (Watts) for wandb logging
        per_cell_power = consumed_power_watts.tolist()
        self.energy_consumption_per_cell.append(per_cell_power)
        
        if self.debug_energy and (self.num_steps <= 20 or self.num_steps % 10 == 0):
            print(
                f"[ES-DEBUG] step={self.num_steps} ts={self.last_timestamp} "
                f"states={[self.cells_states.get(c, None) for c in self.cellList]} "
                f"per_cell_power={np.round(np.array(per_cell_power), 6).tolist()}"
            )

        return rewards_per_drone
    
    @override
    def get_info(self):
        # Pass the calculated costs to the agent
        return {
            'agent_costs': self.latest_individual_costs,
            'ue_counts': getattr(self, 'latest_ue_counts', {})
        }
    
    # ... (Metodi ausiliari invariati: _init_datalake, _fill_datalake, processing, etc.)
    @override
    def _init_datalake_usecase(self):
        grafana_keys = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER",
            "time_grafana": "INTEGER",
            "step": "INTEGER",
            "throughput": "REAL",
            "en_cons": "REAL",
            "rlf": "REAL",
            "on_cost": "REAL",
            "reward": "REAL"
        }

        energy_keys = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER",
            "cellId": "INTEGER",
            "remainingEnergy": "REAL",
            "energyFraction": "REAL"
        }
        self.datalake._create_table("bsState",self.gnb_state_keys)  
        self.datalake._create_table("grafana",grafana_keys)
        self.datalake._create_table("energy",energy_keys)  
        return super()._init_datalake_usecase()

    @override
    def _fill_datalake_usecase(self):
        for file_path in glob.glob(os.path.join(self.sim_path, 'bsState.txt')):
            with open(file_path, 'r') as csvfile:
                # Skip header and parse robustly to tolerate partial tail lines
                # while ns-3 is still writing the file.
                next(csvfile, None)
                for line in csvfile:
                    parts = line.strip().split()
                    # Required columns: Timestamp UNIX Id State
                    if len(parts) < 4:
                        continue

                    try:
                        timestamp = int(parts[1])  # UNIX
                        cell_id = int(parts[2])    # Id
                        state = int(parts[3])      # State
                    except (ValueError, TypeError):
                        continue

                    if timestamp < self.last_timestamp:
                        continue

                    bs_row = {
                        'timestamp': timestamp,
                        'ueImsiComplete': None,
                        'cellId': cell_id,
                        'state': state,
                        'energyFraction': 0.0,
                        'remainingEnergy': 0.0
                    }

                    # Optional columns in bsState.txt:
                    # EnergyFraction RemainingEnergy
                    if len(parts) >= 6:
                        try:
                            bs_row['energyFraction'] = float(parts[4])
                            bs_row['remainingEnergy'] = float(parts[5])
                        except (ValueError, TypeError):
                            pass

                    self.datalake.insert_data("bsState", bs_row)
                    self.last_timestamp = timestamp

    def ue_centric_tocell_centric(self, df):
        df.drop(columns=['ueImsiComplete', 'L3 serving SINR'], inplace=True)
        df = df.drop_duplicates()
        df.reset_index(drop=True, inplace=True)
        return df
        
    def rename_columns(self, columns, cell_no):
        cols = []
        for i in columns:
            cols.append(i+'_'+str(cell_no))
        return cols

    def offline_training_preprocessing(self, df):
        df = self.add_eekpi_qpsk_16_64qam_sum_and_ratio(df)
        df.sort_values(by=["timestamp"], ascending=True, inplace=True)
        df["state"] = df["state"].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))
        cell_df = pd.DataFrame()
        is_initial_cell = True
        for cell in self.cellList:
            temp_cell_df = df.loc[df["nrCellId"] == cell].copy()
            temp_cell_df['RRU_PRBTOTDL'] = (temp_cell_df['RRU.PrbUsedDl'] / 139) * 100
            temp_cell_df['EEKPI_RL'] = (
                temp_cell_df['QosFlow.PdcpPduVolumeDL_Filter'] / temp_cell_df['TB.TotNbrDl.1']
            )
            temp_cell_df.columns = self.rename_columns(temp_cell_df.columns, cell)
            temp_cell_df.rename(columns={f"timestamp_{cell}": "timestamp"}, inplace=True)
            if is_initial_cell:
                cell_df = temp_cell_df
                is_initial_cell = False
            else:
                cell_df = pd.merge(cell_df, temp_cell_df, how="outer", on=["timestamp"])
            del temp_cell_df
        
        es_state_columns = cell_df.columns[cell_df.columns.str.startswith("state_")]
        cell_df[es_state_columns] = cell_df[es_state_columns].fillna(1).astype(np.int64)
        cell_df = cell_df.fillna(0)
        cell_df = self.es_on_cost_calculation(cell_df)
        
        columns_to_clean = {
            'QosFlow.PdcpPduVolumeDL_Filter_': 'float32',
            'TB.TotNbrDl.1_': 'float32',
            'EEKPI_RL_': 'float32',
            'RLF_VALUE': 'float32'
        }
        for prefix, dtype in columns_to_clean.items():
            for col in cell_df.columns[cell_df.columns.str.startswith(prefix)]:
                cell_df.loc[cell_df[col] == '', col] = 0.0
                cell_df[col] = cell_df[col].astype(dtype)
        
        cell_df = cell_df.round(2)
        cell_df['ACTION_BINARY'] = self.previous_inverted_action
        cell_df['ACTION_BINARY'] = cell_df['ACTION_BINARY'].astype(str)
        cell_df['ZERO_COUNT'] = cell_df['ACTION_BINARY'].apply(lambda x: x.count('0'))
        
        kpi_sums = {
            'SUM_QosFlow.PdcpPduVolumeDL_Filter': 'QosFlow.PdcpPduVolumeDL_Filter_',
            'SUM_TB.TotNbrDl.1': 'TB.TotNbrDl.1_',
            'SUM_ES_ON_COST': 'ES_ON_COST_',
            'SUM_RLF_VALUE': 'RLF_VALUE_'
        }
        for sum_col, prefix in kpi_sums.items():
            cell_df[sum_col] = cell_df.filter(like=prefix).sum(axis=1)
        for sum_col in kpi_sums.keys():
            cell_df[sum_col] = pd.to_numeric(cell_df[sum_col])
            
        for cell in self.cellList:
            tb_col = f'TB.TotNbrDl.1_{cell}'
            qos_col = f'QosFlow.PdcpPduVolumeDL_Filter_{cell}'
            eekpi_col = f'EEKPI_RL_{cell}'
            cell_df[tb_col] = cell_df[tb_col].apply(lambda x: x if x != 0 else 0.00001)
            cell_df[eekpi_col] = cell_df.apply(lambda x: x[qos_col] / x[tb_col], axis=1)
        return cell_df
    
    def add_eekpi_qpsk_16_64qam_sum_and_ratio(self, df):
        df['TB.TOTNBRDLINITIAL.SUM'] = (
            df['TB.TotNbrDlInitial.Qpsk'] +
            df['TB.TotNbrDlInitial.16Qam'] +
            df['TB.TotNbrDlInitial.64Qam']
        )
        df['TB_TOTNBRDLINITIAL_64QAM_RATIO'] = (
            df['TB.TotNbrDlInitial.64Qam'] / df['TB.TOTNBRDLINITIAL.SUM']
        ).fillna(0.00001)
        df['RRU.PrbUsedDl'] = df['RRU.PrbUsedDl'].replace(0, 0.00001)
        df['TB.TotNbrDl.1'] = df['TB.TotNbrDl.1'].replace(0, 0.00001)
        return df

    def getRLFCounter(self, df, columns):
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype('Int64')
        sinr = df['L3 serving SINR'].astype(str)
        sinr = (sinr.str.replace('\u2212', '-', regex=False).str.replace('[dD][bB]', '', regex=True).str.strip())
        sinr = sinr.str.extract(r'([-+]?\d*\.?\d+)')[0]
        sinr = pd.to_numeric(sinr, errors='coerce')
        sinr = sinr.replace([np.inf, -np.inf], np.nan)
        df['L3 serving SINR'] = sinr
        df['RLF_Counter'] = 0.0
        df['RLF_VALUE'] = 0
        if 'RLF_Counter' not in columns:
            columns += ['RLF_Counter', 'RLF_VALUE']
        grouped = df.groupby(['timestamp', 'nrCellId'], dropna=False)
        for (timestamp, cell), group in grouped:
            total_count = group['L3 serving SINR'].notna().sum()
            if total_count > 0:
                num_values = (group['L3 serving SINR'] < -5).sum()
                rlf_value = (num_values / total_count) * 100.0
                mask = (df['timestamp'] == timestamp) & (df['nrCellId'] == cell)
                df.loc[mask, 'RLF_Counter'] = rlf_value
                df.loc[mask, 'RLF_VALUE'] = int(num_values)
        return df, columns


    def es_on_cost_calculation(self, cell_df):
        for cell in self.cellList:
            current_timestamp = self.last_timestamp
            time_diff_obs = []
            # bsState convention: state == 0 -> active, state == 1 -> energy-saving.
            current_state_raw = self.cells_states.get(cell, 1)
            current_state = 1 if current_state_raw == 0 else 0
            
            if current_state == 1:
                if self.cell_timestamp_state_dict[cell] == float('inf'):
                    time_diff_obs.append(100)
                    self.cell_timestamp_state_dict[cell] = current_timestamp
                else:
                    time_diff_obs.append(current_timestamp - self.cell_timestamp_state_dict[cell]+100)
            else:
                time_diff_obs.append(float('inf'))
                self.cell_timestamp_state_dict[cell] = float('inf')

            time_diff_obs_col = f'TIME_DIFF_OBS_{cell}'
            cell_df[time_diff_obs_col] = time_diff_obs
            es_on_cost_col = f'ES_ON_COST_{cell}'
            cell_df[es_on_cost_col] = cell_df[time_diff_obs_col].apply(
                lambda diff: self.Cf * ((1 - self.lambdaf) ** (diff * self.time_factor)) if diff != float('inf') else 0
            )
        return cell_df
    
    def bs_states_list(self):
        cell_states_table = self.datalake.read_table('bsState')
        states_of_interest = []
        for cell_state in cell_states_table:
            if cell_state[0] == self.last_timestamp:
                states_of_interest.append(cell_state)
        current_kpms = []
        for state in states_of_interest:
            current_kpms.append(state[3])
        inverted_action_ar = [1 if element == 0 else 0 for element in current_kpms]
        return inverted_action_ar
