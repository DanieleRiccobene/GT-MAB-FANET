from typing_extensions import override
import numpy as np
import pandas as pd
from nsoran.ns_env import NsOranEnv 
import pandas as pd
import glob
import csv
import os
from gymnasium import spaces

# Reward function components
# 'SUM_QOSFLOW_PDCPPDUVOLUMEDL_FILTER' represents the sum of individual QoS flow volume for downlink PDU per cell.
# 'SUM_TB_TOTNBRDL_1' is the total number of downlink transport blocks across cells.
# 'SUM_RLF_VALUE' indicates the total number of radio link failures (RLFs):
# the C++ RLF_Counter KPM plus expected UEs that are absent from KPI rows.
# 'SUM_ES_ON_COST' calculates the total cost associated with energy-saving states.
#     ES_ON_COST is computed using the es_on_cost_calculation() method.
# 'ZERO_COUNT' indicates how many zero states are present, using the zero_count() method.

# Reward function formula:
# 0.51 * SUM_QOSFLOW_PDCPPDUVOLUMEDL_FILTER_QUANTILE
# - 0.19 * (SUM_TB_TOTNBRDL_1_QUANTILE + ZERO_COUNT_QUANTILE)
# - 0.2 * SUM_RLF_VALUE_QUANTILE
# - 0.1 * SUM_ES_ON_COST_QUANTILE

# Base station state attributes used in the model:
# EEKPI_RL_{i} represents the ratio of QoS flow volume to transport blocks for downlink.
# ES_ON_COST_{i} is included as part of the reward function (see es_on_cost_calculation()).
# QOSFLOW_PDCPPDUVOLUMEDL_FILTER_{i} measures the QoS PDU volume for downlink flows.
# RLF_COUNTER_{i} is read from the C++ RLF_Counter KPM.
# RRU_PRBTOTDL_{i} represents the percentage of physical resource blocks used for downlink:
#     df.apply(lambda x: (x['RRU_PRBUSEDDL'] / 139) * 100, axis=1)
# TB_TOTNBRDLINITIAL_64QAM_RATIO_{i} computes the ratio of 64QAM transport blocks:
#     TB_TOTNBRDLINITIAL_SUM = sum of QPSK, 16QAM, and 64QAM initial transport blocks.
#     Ratio is calculated as TB_TOTNBRDLINITIAL_64QAM / TB_TOTNBRDLINITIAL_SUM, handling division by zero.

# Attributes extracted from ns-3 logs per cell:
# - QOSFLOW_PDCPPDUVOLUMEDL_FILTER: QoS flow downlink volume.
# - TB_TOTNBRDL_1: Total number of downlink transport blocks.
# - RLF_Counter: radio link failure counter emitted by ns-3.
# - RRU_PRBUSEDDL: Physical resource block usage for downlink.
# - TB_TOTNBRDLINITIAL_64QAM, TB_TOTNBRDLINITIAL_QPSK, TB_TOTNBRDLINITIAL_16QAM: Transport block metrics by modulation scheme.
# - ES_STATE: Energy-saving state (1 = OFF, 0 = ON).


class EnergySavingEnv(NsOranEnv):
    
    gnb_state_keys = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER",
            "cellId": "INTEGER",
            "state": "INTEGER"
        } 
        
    def __init__(self, ns3_path:str, scenario_configuration:dict, output_folder:str, optimized:bool, do_heuristic:bool = False, skip_configuration:bool = False, skip_build:bool = False):
        super().__init__(ns3_path=ns3_path, scenario='scenario-fanet', scenario_configuration=scenario_configuration,
                output_folder=output_folder, optimized=optimized,
                skip_configuration=skip_configuration, skip_build=skip_build,
                control_header = ['timestamp','cellId','hoAllowed'], log_file='EsActions.txt', control_file='es_actions_for_ns3.csv')
        
        self.folder_name = "Simulation"
        self.ns3_simulation_time = self.scenario_configuration['simTime']*1000
        
        # 1. Numero di droni dinamico
        num_drones = int(self.scenario_configuration.get('nMmWaveEnbNodes', 9))
        self.cellList = [i + 2 for i in range(num_drones)]
        
        # 2. Generazione colonne dinamiche
        base_columns = [
            'EEKPI_RL', 'ES_ON_COST', 'QosFlow.PdcpPduVolumeDL_Filter',
            'RLF_Counter', 'RLF_VALUE', 'RRU_PRBTOTDL', 'RRU.PrbUsedDl',
            'TB_TOTNBRDLINITIAL_64QAM_RATIO'
        ]
        
        self.columns_state = []
        for base_col in base_columns:
            for cell in self.cellList:
                self.columns_state.append(f'{base_col}_{cell}')
                
        self.columns_state.extend([
            'SUM_QosFlow.PdcpPduVolumeDL_Filter',
            'SUM_RLF_VALUE',
            'SUM_TB.TotNbrDl.1',
            'SUM_ES_ON_COST',
            'ZERO_COUNT'
        ])

        self.columns_reward = [
            'SUM_QosFlow.PdcpPduVolumeDL_Filter',
            'SUM_TB.TotNbrDl.1',
            'SUM_RLF_VALUE',
            'SUM_ES_ON_COST',
            'ZERO_COUNT'
        ]
        
        self.action_list = [i for i in range(2 ** len(self.cellList))]
        self.observations = []
        self.cells_states = {}
        self.columns_energy = [f'remainingEnergy_{c}' for c in self.cellList] + [f'energyFraction_{c}' for c in self.cellList]
        self.columns_state = self.columns_state + self.columns_energy
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.columns_state),), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action_list))
        self.previous_energy_joules = None
        
        self.cell_timestamp_state_dict = {cell: float('inf') for cell in self.cellList} 
        self.Cf = 1
        self.lambdaf = 0.1
        self.time_factor = 0.01
        self.heur = do_heuristic
        self.num_steps = 0
        self.previous_inverted_action = "0" * len(self.cellList)
        self.debug_energy = os.getenv("ES_DEBUG_ENERGY", "0") == "1"

        self.avg_qos = []
        self.avg_nrdbl = []
        self.avg_rlf = []
        self.avg_unconnected_ues = []
        self.average_energy_consumption = []
        self.energy_consumption_per_cell = []
        self.crashed = False
        self.latest_connected_ues = 0
        self.latest_unconnected_ues = 0
    
    def _reset_stats(self):
        self.avg_qos = []
        self.avg_nrdbl = []
        self.avg_rlf = []
        self.avg_unconnected_ues = []
        self.average_energy_consumption = []
        self.crashed = False
        self.energy_consumption_per_cell = []
        self.previous_energy_joules = None
        self.latest_connected_ues = 0
        self.latest_unconnected_ues = 0

    def _expected_total_ues(self):
        try:
            return int(self.scenario_configuration.get('ues', 0))
        except (TypeError, ValueError):
            return 0

    def _update_unconnected_ue_count(self, df):
        expected_ues = self._expected_total_ues()
        connected_ues = int(df['ueImsiComplete'].dropna().nunique()) if 'ueImsiComplete' in df else 0
        self.latest_connected_ues = connected_ues
        self.latest_unconnected_ues = max(expected_ues - connected_ues, 0)
        return self.latest_unconnected_ues

    @override
    def _compute_action(self, action):
        """ Function that converts the agents action defined in Gym into the ns-O-RAN required format according to the use case
        """
        #print("action:", action)
        #print("self.heur:", self.heur)
        cell_act_comb_lst = []
        if self.heur == True:
            cell_id=[2, 3, 4, 5, 6, 7, 8, 9, 10]
            cell_act_comb_lst=[[cell,act] for cell,act in zip(cell_id,action)]
        else:
            # Fallback per compatibilità (se action fosse un intero)
            dec_action = self.action_list[action]
            bin_format = f'0{len(self.cellList)}b'
            bin_actions = [int(i) for i in list(format(dec_action, bin_format))]
            cell_act_comb_lst = []
            self.previous_inverted_action = "".join(["0" if b == 1 else "1" for b in bin_actions])
            for cell, bin_val in zip(self.cellList, bin_actions):
                # ns-3 stores hoAllowed: 1 = active/available, 0 = energy-saving.
                ns3_val = 1 if bin_val == 1 else 0
                cell_act_comb_lst.append([cell, ns3_val])
        return cell_act_comb_lst
    
    def _update_cell_states(self):
        """Function that updates the states of the cells saved in a class variable
        """
        target_ts = self.last_timestamp - 100
        cell_states_table = self.datalake.read_rows_at_timestamp('bsState', target_ts)
        latest_states = {}
        for cell_state in cell_states_table:
            if len(cell_state) >= 4 and cell_state[0] == target_ts:
                latest_states[int(cell_state[2])] = int(cell_state[3])

        if not self.cells_states:
            self.cells_states = {cell_id: 1 for cell_id in self.cellList}

        for cell_id in self.cellList:
            if cell_id in latest_states:
                self.cells_states[cell_id] = latest_states[cell_id]

    @override
    def _get_obs(self):
        # Database state: 1 = active/available, 0 = energy-saving/unavailable.
        kpms_raw = ["nrCellId", "QosFlow.PdcpPduVolumeDL_Filter", "TB.TotNbrDl.1", "RLF_Counter", "RRU.PrbUsedDl", "TB.TotNbrDlInitial.64Qam", "TB.TotNbrDlInitial.Qpsk", "TB.TotNbrDlInitial.16Qam"]  
        #print("last timestamp:", self.last_timestamp)     
        ue_kpms = self.datalake.read_kpms(self.last_timestamp, kpms_raw) 
        #print("ue kpms pre cell states update:", ue_kpms)
        self._update_cell_states()  
        #print("cell states updated")
        # Now cells_states is updated with state of latest cells           
        # iterate over ue_kpms to add state value
        ue_complete_kpms = []
        # For each row in ue_kpms look for its state into cells_states and save it
        #print("ue_kpms =", ue_kpms)
        if ue_kpms is None or len(ue_kpms) == 0:
            print("No UE KPMs found for the last timestamp. Marking environment as crashed.")
            self.crashed = True
            self.latest_connected_ues = 0
            self.latest_unconnected_ues = self._expected_total_ues()
            self.observations = pd.DataFrame(
                [np.zeros(len(self.columns_state), dtype=np.float32)],
                columns=self.columns_state,
            )
            if 'SUM_RLF_VALUE' in self.observations:
                self.observations['SUM_RLF_VALUE'] = float(self.latest_unconnected_ues)
            return self.observations.iloc[0].values
        else:
            for ue_kpm in ue_kpms:
                # Create a new tuple with the same elements of ue_kpm + state in the latest position
                # State is calculated using self.cell_state[ue_kpm[1]] because index 0 if for ueIMSI
                state = self.cells_states.get(ue_kpm[1], ())  # Get state using cell_id (second element in ue_kpm)
                # Create a new tuple by concatenating ue_kpm with the state tuple
                new_ue_kpm = ue_kpm + (state,)
                ue_complete_kpms.append(new_ue_kpm)   
        # At this point observations_raw is UEIMSI + kpms_raw + state
        # Define the column names based on kpms_raw and the single state column
        columns = ['ueImsiComplete'] + kpms_raw + ['state']  # Add 'state' as the last column
        # Create the DataFrame
        df = pd.DataFrame(ue_complete_kpms, columns=columns)
        df["timestamp"] = self.last_timestamp
        self._update_unconnected_ue_count(df)
        # Count the RLF at UEs level
        df, columns= self.getRLFCounter(df, columns)
        # Now we need to convert the dataframe from UEs centric to Cell centric
        df = self.ue_centric_tocell_centric(df)
        self.observations = self.offline_training_preprocessing(df)
        

        bs_rows = self.datalake.read_latest_bsstate_before(self.last_timestamp, self.cellList)
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
        energy_data = {}
        for cell in self.cellList:
            if cell in latest_by_cell:
                _, rem, frac = latest_by_cell[cell]
                energy_data[f'remainingEnergy_{cell}'] = np.float32(rem)
                energy_data[f'energyFraction_{cell}'] = np.float32(frac)
            else:
                energy_data[f'remainingEnergy_{cell}'] = np.float32(0.0)
                energy_data[f'energyFraction_{cell}'] = np.float32(0.0)
        
        self.observations = pd.concat([self.observations, pd.DataFrame([energy_data])], axis=1)
        states = self.observations[self.columns_state].iloc[0].values
        return states
        
    @override
    def _compute_reward(self):
        self.num_steps += 1
        cell_df = self.observations[self.columns_reward].copy()
        db_row = {}
        db_row['timestamp'] = self.last_timestamp
        db_row['ueImsiComplete'] = None
        db_row['time_grafana'] = self.last_timestamp
        db_row['step'] = self.num_steps
        db_row['throughput'] = float(cell_df['SUM_QosFlow.PdcpPduVolumeDL_Filter'].iloc[0]) * 10 / 10**6
        db_row['en_cons'] = float(cell_df['SUM_TB.TotNbrDl.1'].iloc[0])
        db_row['rlf'] = float(cell_df['SUM_RLF_VALUE'].iloc[0])
        db_row['on_cost'] = float(cell_df['SUM_ES_ON_COST'].iloc[0])

        # Step 2: Calculate reward values
        cell_df['reward'] = (
            0.7 * cell_df['SUM_QosFlow.PdcpPduVolumeDL_Filter'].iloc[0]
            - 0.1 * cell_df['SUM_RLF_VALUE'].iloc[0]
            - 0.1 * cell_df['SUM_TB.TotNbrDl.1'].iloc[0]
            - 0.1 * cell_df['ZERO_COUNT'].iloc[0]
            - 0.1 * cell_df['SUM_ES_ON_COST'].iloc[0]
        )

        energy_joules_list = []
        for cell in self.cellList:
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

        penalty_battery = 0.0
        for idx, cell in enumerate(self.cellList):
            # La penalità è basata ESCLUSIVAMENTE sul consumo di questo specifico drone
            # Il peso (es. -0.001) va calibrato in base all'ordine di grandezza dei Watt
            penalty_battery += float(-0.001 * consumed_energy_joules[idx])
        
        reward = cell_df['reward'][0] + penalty_battery
        # Grafana db 
        db_row['reward'] = reward
        # Insert the data into the datalake
        self.datalake.insert_data("grafana", db_row)

        self.avg_qos.append(cell_df['SUM_QosFlow.PdcpPduVolumeDL_Filter'].iloc[0])
        self.avg_nrdbl.append((cell_df['SUM_TB.TotNbrDl.1'] + cell_df['ZERO_COUNT']).iloc[0])
        self.avg_rlf.append(cell_df['SUM_RLF_VALUE'].iloc[0])
        self.avg_unconnected_ues.append(float(self.latest_unconnected_ues))
        self.average_energy_consumption.append(cell_df['SUM_ES_ON_COST'].iloc[0])
        
        per_cell_es = [
            float(self.observations[f'ES_ON_COST_{c}'].iloc[0])
            if f'ES_ON_COST_{c}' in self.observations else 0.0
            for c in self.cellList
        ]
        self.energy_consumption_per_cell.append(per_cell_es)
        #print("reward components: ", cell_df['SUM_QosFlow.PdcpPduVolumeDL_Filter'], cell_df['SUM_TB.TotNbrDl.1'], cell_df['ZERO_COUNT'], cell_df['SUM_RLF_VALUE'], cell_df['SUM_ES_ON_COST'])
        return reward
    
    

    @override
    def _init_datalake_usecase(self):
        # Grafana table
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
        self.datalake._create_table("bsState",self.gnb_state_keys)  
        self.datalake._create_table("grafana",grafana_keys)  
        return super()._init_datalake_usecase()

    @override
    def _fill_datalake_usecase(self):
        for file_path in glob.glob(os.path.join(self.sim_path, 'bsState.txt')):
            with open(file_path, 'r') as csvfile:
                for row in csv.DictReader(csvfile, delimiter=' '):
                    if row.get('UNIX') is None:
                        continue

                    try:
                        timestamp = int(row['UNIX'])
                        if timestamp >= self.last_timestamp:
                            db_row = {}
                            db_row['timestamp'] = timestamp
                            db_row['ueImsiComplete'] = None  # Set to null
                            db_row['cellId'] = int(row['Id'])
                            db_row['state'] = int(row['State'])
                            # Insert the data into the datalake
                            self.datalake.insert_data("bsState", db_row)
                            # Update the last timestamp
                            self.last_timestamp = timestamp
                    except (ValueError, TypeError):
                        # Salta righe malformate o incomplete
                        continue

    def ue_centric_tocell_centric(self, df):
        """Function used to clean the dataframe with ns-3 row data
        """
        # Delete columns that are ue centric
        df.drop(columns=['ueImsiComplete', 'L3 serving SINR'], inplace=True, errors='ignore')
        # Remove completely identical rows
        df = df.drop_duplicates()
        # Reset index
        df.reset_index(drop=True, inplace=True)
        return df
        
    def rename_columns(self, columns, cell_no):
        cols = []
        for i in columns:
            cols.append(i+'_'+str(cell_no))
        return cols

    def offline_training_preprocessing(self, df):
        """
        Preprocess the DataFrame by calculating KPIs and KPMs for reward for each cell.
        """
        df = self.add_eekpi_qpsk_16_64qam_sum_and_ratio(df)
        # Sort the final DataFrame by the TIMESTAMP column in ascending order
        df.sort_values(by=["timestamp"], ascending=True, inplace=True)
        # Invert State values
        df["state"] = df["state"].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))
        # Initialize an empty DataFrame to store the merged results
        cell_df = pd.DataFrame()
        is_initial_cell = True  # Flag to track the first cell's DataFrame
        # Iterate over the list of cells
        for cell in self.cellList:
            # Filter the data for the current cell and create a copy to avoid modifying the original DataFrame
            temp_cell_df = df.loc[df["nrCellId"] == cell].copy()
            # Calculate the percentage of PRB utilization
            temp_cell_df['RRU_PRBTOTDL'] = (temp_cell_df['RRU.PrbUsedDl'] / 139) * 100
            # Calculate the general EEKPI (Energy Efficiency KPI) for the downlink
            temp_cell_df['EEKPI_RL'] = (
                temp_cell_df['QosFlow.PdcpPduVolumeDL_Filter'] / temp_cell_df['TB.TotNbrDl.1']
            )
            # Rename the columns for the current cell to ensure uniqueness
            temp_cell_df.columns = self.rename_columns(temp_cell_df.columns, cell)
            # Rename the TIMESTAMP column to align across all cells for merging
            temp_cell_df.rename(columns={f"timestamp_{cell}": "timestamp"}, inplace=True)
            # Merge the data of the current cell with the overall DataFrame
            if is_initial_cell:
                cell_df = temp_cell_df
                is_initial_cell = False  # Mark the first cell as processed
            else:
                cell_df = pd.merge(cell_df, temp_cell_df, how="outer", on=["timestamp"])
            # Free up memory by deleting the temporary DataFrame
            del temp_cell_df
        # Replace NaN values in 'ES_STATE' columns with 1 and convert to integers
        es_state_columns = cell_df.columns[cell_df.columns.str.startswith("state_")]
        cell_df[es_state_columns] = cell_df[es_state_columns].fillna(1).astype(np.int64)
        # Replace all remaining NaN values in the DataFrame with 0
        cell_df = cell_df.fillna(0)
        # Step 1: Calculate ES on-costs for all cells
        cell_df = self.es_on_cost_calculation(cell_df)
        # Step 2: Save BS states
        # Step 3: Manage missing values for specific columns
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
        
        # Step 4: Round numeric columns to 2 decimal places
        cell_df = cell_df.round(2)
        # Step 5: Calculate reward components
        # Convert decimal to binary string (padded to 7 bits)
        cell_df['ACTION_BINARY'] = self.previous_inverted_action
        cell_df['ACTION_BINARY'] = cell_df['ACTION_BINARY'].astype(str)
        # Count the number of zeros in the binary representation
        cell_df['ZERO_COUNT'] = cell_df['ACTION_BINARY'].apply(
            lambda x: x.count('0')
        )
        # Sum columns for KPIs and costs
        kpi_sums = {
            'SUM_QosFlow.PdcpPduVolumeDL_Filter': 'QosFlow.PdcpPduVolumeDL_Filter_',
            'SUM_TB.TotNbrDl.1': 'TB.TotNbrDl.1_',
            'SUM_ES_ON_COST': 'ES_ON_COST_',
            'SUM_RLF_VALUE': 'RLF_VALUE_'
        }
        for sum_col, prefix in kpi_sums.items():
            cell_df[sum_col] = cell_df.filter(like=prefix).sum(axis=1)
        cell_df['SUM_RLF_VALUE'] = (
            cell_df['SUM_RLF_VALUE'] + float(getattr(self, 'latest_unconnected_ues', 0))
        )
        # Ensure numeric type for summed columns
        for sum_col in kpi_sums.keys():
            cell_df[sum_col] = pd.to_numeric(cell_df[sum_col])
        # Step 6: Calculate EEKPI_RL by cell
        for cell in self.cellList:
            tb_col = f'TB.TotNbrDl.1_{cell}'
            qos_col = f'QosFlow.PdcpPduVolumeDL_Filter_{cell}'
            eekpi_col = f'EEKPI_RL_{cell}'
            # Avoid division by zero
            cell_df[tb_col] = cell_df[tb_col].apply(
                lambda x: x if x != 0 else 0.00001
            )
            # Calculate EEKPI_RL
            cell_df[eekpi_col] = cell_df.apply(
                lambda x: x[qos_col] / x[tb_col], axis=1
            )
        cell_df = cell_df.copy()
        return cell_df
    
    def add_eekpi_qpsk_16_64qam_sum_and_ratio(self, df):
        """
        Adds multiple EEKPI-related columns to the DataFrame by performing operations on existing columns.
        """
        # Calculate the total sum of TB_TOTNBRDLINITIAL_* columns
        df['TB.TOTNBRDLINITIAL.SUM'] = (
            df['TB.TotNbrDlInitial.Qpsk'] +
            df['TB.TotNbrDlInitial.16Qam'] +
            df['TB.TotNbrDlInitial.64Qam']
        )
        df['TB_TOTNBRDLINITIAL_64QAM_RATIO'] = (
            df['TB.TotNbrDlInitial.64Qam'] / df['TB.TOTNBRDLINITIAL.SUM']
        ).fillna(0.00001)
        # Handle RRU_PRBUSEDDL (avoid division by zero by using a small default value)
        df['RRU.PrbUsedDl'] = df['RRU.PrbUsedDl'].replace(0, 0.00001)
        # TB.TotNbrDl.1 handling to avoid division by zero
        df['TB.TotNbrDl.1'] = df['TB.TotNbrDl.1'].replace(0, 0.00001)
        return df

    def getRLFCounter(self, df, columns):
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').astype('Int64')

        if 'RLF_Counter' not in df.columns:
            df['RLF_Counter'] = 0.0
        df['RLF_Counter'] = pd.to_numeric(df['RLF_Counter'], errors='coerce').fillna(0.0)
        df['RLF_Counter'] = df['RLF_Counter'].clip(lower=0.0)
        df['RLF_VALUE'] = 0.0
        if 'RLF_Counter' not in columns:
            columns.append('RLF_Counter')
        if 'RLF_VALUE' not in columns:
            columns.append('RLF_VALUE')

        grouped = df.groupby(['timestamp', 'nrCellId'], dropna=False)
        for (timestamp, cell), group in grouped:
            rlf_values = group['RLF_Counter'].dropna()
            if rlf_values.empty:
                continue

            unique_values = rlf_values.unique()
            if len(unique_values) == 1 and unique_values[0] > 1.0:
                rlf_value = float(unique_values[0])
            else:
                rlf_value = float(rlf_values.sum())

            mask = (df['timestamp'] == timestamp) & (df['nrCellId'] == cell)
            df.loc[mask, 'RLF_Counter'] = rlf_value
            df.loc[mask, 'RLF_VALUE'] = rlf_value

        return df, columns


    def es_on_cost_calculation(self, cell_df):
        """
        Calculate the energy-saving (ES) on-cost for each cell in the cell list.
        Cost increases if cell turning OFF too fast (cell stays ON (state=0) for too short)
        Intent is to AVOID turning OFF the same cell too frequently (1/ES MODE , ON->OFF->ON) 
        """
        for cell in self.cellList:
            current_timestamp = self.last_timestamp
            # Initialize a list for TIME_DIFF_OBS
            time_diff_obs = []
            
            # bsState stores hoAllowed: 1 -> active, 0 -> energy-saving.
            current_state_raw = self.cells_states.get(cell, 1)
            current_state = 1 if current_state_raw == 0 else 0
            
            if current_state == 1:
                # If state == 1, calculate time difference from saved timestamp
                if self.cell_timestamp_state_dict[cell] == float('inf'):
                    # First time state becomes 1, set initial timestamp
                    time_diff_obs.append(100)  # No time has elapsed yet
                    # Update the saved timestamp for this cell
                    self.cell_timestamp_state_dict[cell] = current_timestamp
                else:
                    # Calculate time difference
                    time_diff_obs.append(current_timestamp - self.cell_timestamp_state_dict[cell]+100)
            else:
                # If state == 0, reset the saved timestamp and set time difference to inf
                time_diff_obs.append(float('inf'))
                self.cell_timestamp_state_dict[cell] = float('inf')

            # Add TIME_DIFF_OBS column to the DataFrame
            time_diff_obs_col = f'TIME_DIFF_OBS_{cell}'
            cell_df[time_diff_obs_col] = time_diff_obs
            
            # Calculate the ES on-cost using the formula
            es_on_cost_col = f'ES_ON_COST_{cell}'
            cell_df[es_on_cost_col] = cell_df[time_diff_obs_col].apply(
                lambda diff: self.Cf * ((1 - self.lambdaf) ** (diff * self.time_factor)) if diff != float('inf') else 0
            )
        return cell_df
    
    def bs_states_list(self):
        """The function retrieves the current state of BSs from a datalake table for the latest timestamp 
        and returns a list of corresponding KPMs in an inverted binary format.  
        """
        # Get actual bs state with present timestamp
        cell_states_table = self.datalake.read_rows_at_timestamp('bsState', self.last_timestamp)
        states_of_interest = []
        # Filter rows only from last timestamp
        for cell_state in cell_states_table:
            if cell_state[0] == self.last_timestamp:
                states_of_interest.append(cell_state)
        current_kpms = []
        for state in states_of_interest:
            current_kpms.append(state[3])
        # "cell_state  cellId: State {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}" 
        # Invert 0 and 1 because of logic (0 = ES OFF/Cell ON, 1 = ES ON/Cell OFF)
        inverted_action_ar = [1 if element == 0 else 0 for element in current_kpms]
        return inverted_action_ar
        
