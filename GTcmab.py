import numpy as np
import random

class MultiAgentGameMAB:
    def __init__(self, num_agents=7, epsilon_start=1.0, epsilon_end=0.01, decay_steps=1000):
        self.num_agents = num_agents
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.t = 0  # timestep globale
        
        # --- MODIFICA CRITICA: Dizionari invece di np.zeros ---
        # Poiché usiamo vettori (tuple) come chiavi, non possiamo usare matrici numpy fisse.
        # Q-Tables: Lista di dizionari per ogni agente.
        # Key: Tupla contesto (es. (0, 1, 0...)), Value: np.array([Q_OFF, Q_ON])
        self.q_tables = [{} for _ in range(num_agents)]
        
        # Conteggi per l'aggiornamento alpha
        self.action_counts = [{} for _ in range(num_agents)]

        # Variabili per memorizzare il contesto dell'ultima negoziazione (per l'update)
        # Uniformato il nome a 'last_context_vectors' per coerenza con il metodo update
        self.last_context_vectors = [None] * num_agents
        self.last_actions = [0] * num_agents

    def current_epsilon(self):
        # Decadimento esponenziale dell'esplorazione
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-self.t / self.decay_steps)

    def get_context_vector(self, agent_idx, current_actions):
        """
        Restituisce il vettore di stato degli ALTRI agenti.
        """
        # Creiamo una lista degli stati degli altri escludendo l'agente corrente
        others_states = [current_actions[i] for i in range(self.num_agents) if i != agent_idx]
        return others_states

    def get_q_values(self, agent_idx, context_vector):
        """
        Helper per recuperare i Q-Values dal dizionario in modo sicuro.
        """
        # Le liste non possono essere chiavi di dizionario, usiamo una tupla
        state_key = tuple(context_vector)
        
        # Se la chiave non esiste, inizializza con array di zeri
        if state_key not in self.q_tables[agent_idx]:
            self.q_tables[agent_idx][state_key] = np.zeros(2)
            self.action_counts[agent_idx][state_key] = np.zeros(2)
            
        return self.q_tables[agent_idx][state_key]

    def negotiate(self, previous_actions, cost_vector,max_steps=10):
        """
        Esegue il loop di Best Response Dynamics.
        """
        self.t += 1
        eps = self.current_epsilon()
        
        # Importante: Copiare l'azione precedente per non modificare il riferimento originale
        if isinstance(previous_actions, np.ndarray):
            current_actions = previous_actions.copy()
        else:
            current_actions = list(previous_actions)
        
        # Assicuriamoci che sia un tipo mutabile (lista) o array per gli aggiornamenti
        # Qui lavoriamo con interi 0/1
        
        converged = False
        
        # LOOP DI NEGOZIAZIONE (Game Theory)
        for step in range(max_steps):
            changes = 0

            # Ordine casuale per evitare cicli
            agent_order = np.random.permutation(self.num_agents)
            
            for agent_idx in agent_order:
                # 1. Ottieni il contesto (vettore degli altri)
                ctx_vector = self.get_context_vector(agent_idx, current_actions)
                
                # Selezione Azione: Epsilon-Greedy
                if random.random() < eps:
                    selected_action = random.choice([0, 1])
                else:
                    # Sfruttamento
                    q_values = np.asarray(self.get_q_values(agent_idx, ctx_vector), dtype=np.float64)

                    # Robustness: invalid costs/values can produce NaN utilities and empty candidates.
                    raw_cost = cost_vector[agent_idx] if agent_idx < len(cost_vector) else 0.0
                    base_cost = float(raw_cost) if np.isfinite(raw_cost) else 0.0
                    action_costs = np.full(shape=2, fill_value=base_cost, dtype=np.float64)

                    prev_action = int(previous_actions[agent_idx])
                    if prev_action in (0, 1):
                        # Do not penalize keeping the previous action.
                        action_costs[prev_action] = 0.0

                    net_utilities = q_values - action_costs
                    finite_idx = np.where(np.isfinite(net_utilities))[0]
                    if finite_idx.size == 0:
                        # Fallback to previous action if utilities are invalid.
                        selected_action = prev_action if prev_action in (0, 1) else 0
                    else:
                        finite_vals = net_utilities[finite_idx]
                        max_val = np.max(finite_vals)
                        # isclose avoids precision issues when selecting argmax ties.
                        candidates = finite_idx[np.isclose(finite_vals, max_val)]
                        if candidates.size == 0:
                            selected_action = int(finite_idx[np.argmax(finite_vals)])
                        else:
                            selected_action = int(np.random.choice(candidates))
                
                # Se l'azione cambia rispetto a prima, segniamo il cambiamento
                # Nota: current_actions potrebbe contenere float se viene da np.ones/zeros
                if int(current_actions[agent_idx]) != selected_action:
                    current_actions[agent_idx] = selected_action
                    changes += 1
            
            # Se nessuno cambia idea, equilibrio raggiunto
            if changes == 0:
                converged = True
                break
        
        # Salviamo il contesto finale per l'update
        for i in range(self.num_agents):
            ctx_vector = self.get_context_vector(i, current_actions)
            self.last_context_vectors[i] = tuple(ctx_vector)
            self.last_actions[i] = int(current_actions[i])
            
        return current_actions, converged
    '''
    def update(self, reward):
        """
        Aggiorna le Q-Tables.
        """
        for i in range(self.num_agents):
            # Recuperiamo la tupla di contesto salvata
            ctx_key = self.last_context_vectors[i]
            act = self.last_actions[i]
            
            if ctx_key is None:
                continue

            # Assicuriamoci che l'entry esista (per sicurezza)
            if ctx_key not in self.action_counts[i]:
                 self.action_counts[i][ctx_key] = np.zeros(2)
                 self.q_tables[i][ctx_key] = np.zeros(2)

            # Aggiornamento conteggi
            self.action_counts[i][ctx_key][act] += 1
            
            # Calcolo Alpha (Learning Rate 1/N)
            alpha = 1.0 / self.action_counts[i][ctx_key][act]
            
            # Q-Learning update
            old_val = self.q_tables[i][ctx_key][act]
            self.q_tables[i][ctx_key][act] = old_val + alpha * (reward - old_val)
    '''
    def update(self, rewards_dict, cell_list):
        """
        Aggiorna le Q-Tables usando il dizionario delle reward.
        """
        for i, cell_id in enumerate(cell_list):
            ctx_key = self.last_context_vectors[i]
            act = self.last_actions[i]
            
            if ctx_key is None:
                continue

            if ctx_key not in self.action_counts[i]:
                    self.action_counts[i][ctx_key] = np.zeros(2)
                    self.q_tables[i][ctx_key] = np.zeros(2)

            self.action_counts[i][ctx_key][act] += 1
            alpha = 1.0 / self.action_counts[i][ctx_key][act]
            
            # Estraiamo la reward per questo specifico drone
            # rewards_dict[cell_id] è un array [reward_globale, penalità_individuale]
            agent_reward = float(rewards_dict[cell_id][0] + rewards_dict[cell_id][1])

            old_val = self.q_tables[i][ctx_key][act]
            self.q_tables[i][ctx_key][act] = old_val + alpha * (agent_reward - old_val)
