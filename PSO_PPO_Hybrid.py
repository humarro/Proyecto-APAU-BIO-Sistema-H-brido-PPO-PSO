import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import time

# Declaración del crítico compartido (centralizado)
class Critic(nn.Module):
    def __init__(self, state_dim=2, hidden=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, state):
        return self.net(state)
    
# Declaración del actor distribuido (esto corresponde con cada partícula del ejambre)
class ActorParticle(nn.Module):
    def __init__(self, state_dim=2, action_dim=1, hidden=64):
        super(ActorParticle, self).__init__()

        # La red calcula la media de la acción, por eso su salida entre -1 y 1, el rango del motor
        self.mu_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )
        # Parámetro que correponde con la desviación estandar
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Variables específicas de PSO, cada partícula con su velocidad y su mejor posición
        self.velocity = [torch.zeros_like(p.data) for p in self.parameters()]
        self.pbest_params = [p.data.clone() for p in self.parameters()]
        self.pbest_fitness = -np.inf 

    def forward(self, state):
        mu = self.mu_net(state)
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def get_action(self, state):
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        action = dist.sample()
        action_clipped = torch.clamp(action, -1.0, 1.0) # La acción limitada dentro del rango de movimiento
        return action_clipped, dist.log_prob(action), dist.entropy()
    
    def evaluate(self, state, action):
        # PPO lo usa para evaluar acciones pasadas durante la actualización 
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


# El orquestador híbrido PPO + PSO
class PSO_PPO_Hybrid:
    def __init__(self, num_particles=4, env_name="MountainCarContinuous-v0", use_pso = True, use_ppo = True):
        self.env_name = env_name
        self.num_particles = num_particles
        self.use_pso = use_pso #Interruptor de la ablación (solo PPO)
        self.use_ppo = use_ppo #Interruptor de la ablación (solo PSO)

        self.critic = Critic(state_dim=2) # Un solo crítico para todos
        # Muchas partículas (coches) para explorar el entorno 
        self.swarm = [ActorParticle(state_dim=2, action_dim=1) for _ in range(num_particles)]
        
        # Memoria global del enjambre PSO
        self.gbest_params = None
        self.gbest_fitness = -np.inf
    

    # En esta función cada partícula recoge sus trayectorias de forma independiente,
    # interactuando unicamente con su propia copia del entorno.
    def collect_trajectory(self, env, actor, max_steps=1000, seed_val = None):
        states, actions, log_probs, rewards, dones = [], [], [], [], []

        # Se aplica la semilla si se pasa, solo en el episodio 0 para la inicializacion
        if seed_val is not None:
            state, _ = env.reset(seed=seed_val)
        else:
            state, _ = env.reset()


        for _ in range(max_steps):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_t, log_prob, _ = actor.get_action(state_t)
            
            # Formateamos la acción para que coincida con el entorno 
            action_np = action_t.numpy()[0] 
            next_state, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action_np)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            dones.append(done)
            
            state = next_state
            if done: break
                
        return states, actions, log_probs, rewards, dones
    
    # Función para la actualizacion de la ventaja
    # Aquí es donde vemos el papel clave del Crítico compartido:
    # Al pasar los estados por el crítico, evaluamos las trayectorias de todas 
    # las partículas usando el mismo crítico global. De esta forma se estabiliza el GAE
    def compute_gae(self, states, rewards, dones, gamma=0.99, lam=0.95):
        
        states_t = torch.FloatTensor(np.array(states))

        # El crítico nos proporciona cuánto valor espera de los estados
        with torch.no_grad():
            values = self.critic(states_t).squeeze(-1).numpy()
        
        # Le asignamos el valor 0 al estado terminal
        values = np.append(values, 0.0) 
        advantages = np.zeros(len(rewards))
        gae = 0

        # Recorremos la trayectoria hacia atrás para calcular la ventaja acumulada
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] * (1 - dones[t])

            # TD Error: Recompensa real + Valor futuron - Valor predicho
            delta = rewards[t] + gamma * next_val - values[t]

            # Suavizado exponencial (GAE)
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae
            
        returns = advantages + values[:-1] # Devolvemos la ventaja + valor predicho
        return advantages, returns
    

    # La actualización de las partículas de PSO se hace moviendo los tensores de los pesos de la NN
    def update_particle_pso(self, particle, w=0.7, c1=1.5, c2=1.5):

        # Si no hay mejor global no hacemos nada
        if self.gbest_params is None: return 

        with torch.no_grad():
            for i, param in enumerate(particle.parameters()):

                # Generamos números aleatorios para la estocasticidad 
                r1, r2 = torch.rand_like(param.data), torch.rand_like(param.data)

                # Calculamos las 3 fuerzas de la ecuación de PSO
                inertia = w * particle.velocity[i]
                cognitive = c1 * r1 * (particle.pbest_params[i] - param.data)
                social = c2 * r2 * (self.gbest_params[i] - param.data)
                
                # Actualizamos la velocidad 
                particle.velocity[i] = inertia + cognitive + social
                param.data.add_(particle.velocity[i])

    # En la actualización de PPO usamos gradiente descendente 
    # Este se encarga de la exploración local (aprender a afinar la conducción)
    def ppo_update(self, actor, actor_optimizer, critic_optimizer, 
                   states, actions, old_log_probs, advantages, returns, 
                   clip_eps=0.2, epochs=4, batch_size=64, entropy_coeff=0.01):
        
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions))
        old_lp_t = torch.FloatTensor(np.array(old_log_probs))
        adv_t = torch.FloatTensor(np.array(advantages))
        returns_t = torch.FloatTensor(np.array(returns))
        
        # Normalizamos las ventajas para garantizar la estabilidad del gradiente
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        dataset_size = len(states)

        # Repetimos la actualización varias veces 
        for _ in range(epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                idx = indices[start:start + batch_size]
                b_states, b_actions, b_old_lp = states_t[idx], actions_t[idx], old_lp_t[idx]
                b_adv, b_returns = adv_t[idx], returns_t[idx]

                # Evaluamos que probabilidad le daría la red actual a las acciones pasadas
                new_lp, entropy = actor.evaluate(b_states, b_actions)
                values = self.critic(b_states).squeeze(-1)

                # Ratio de PPO (porbabilidad nueva / probabilidad vieja)
                ratio = torch.exp(new_lp - b_old_lp)

                # Función objetivo recortada
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Pérdida del crítico (usamos error cuadrático medio)
                critic_loss = F.mse_loss(values, b_returns)

                # Actualización del actor (Política)
                actor_total_loss = actor_loss - entropy_coeff * entropy.mean()
                actor_optimizer.zero_grad()
                actor_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_optimizer.step()
                
                # Actualización del crítico compartido
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                critic_optimizer.step()
    
    # El bucle train define cuándo se interactúa, cuándo se optimiza por PPO y cuándo se explora por PSO
    def train(self, iterations=150, seed=42, clip_eps=0.2, w=0.7, c1=1.5, c2=1.5):
        
        # Fijamos la semilla para reproducibilidad de los experimentos
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Creamos una instancia del entorno para cada partícula (simulación en paralelo)
        envs = [gym.make(self.env_name) for _ in range(self.num_particles)]

        # Creamos los optimizadores: uno para el Crítico y una lista para los Actores (coches)
        critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=3e-4) for actor in self.swarm]
        
        # Lista para almacenar los registros de esta ejecución
        logs = []
        start_time_total = time.time()
        time_spent_pso = 0.0

        total_timesteps = 0 # Contador de pasos para la eficiencia muestral 
        
        # Bucle principal de entrenamiento
        for it in range(iterations):
            trajectories = []
            mean_population_fitness = 0
            
            # Recolección y evaluación de aptitud: PSO - memoria
            for i, actor in enumerate(self.swarm):
                # Pasamos la semilla SOLO en la iteración 0 para inicializar el entorno
                env_seed = seed if it == 0 else None 
                states, actions, log_probs, rewards, dones = self.collect_trajectory(envs[i], actor, seed_val=env_seed)
                
                # La aptitud es la suma de las recompensas 
                fitness = sum(rewards)
                mean_population_fitness += fitness
                total_timesteps += len(rewards) 

                # Actualizamos la memoria del PSO (mejor personal)
                if fitness > actor.pbest_fitness:
                    actor.pbest_fitness = fitness
                    actor.pbest_params = [p.data.clone() for p in actor.parameters()]

                    # Actualizamos la memoria del PSO (mejor global)
                    if fitness > self.gbest_fitness:
                        self.gbest_fitness = fitness
                        self.gbest_params = [p.data.clone() for p in actor.parameters()]
                
                trajectories.append((states, actions, log_probs, rewards, dones))
                
            mean_population_fitness /= self.num_particles
                

            # Exploración local: PPO - gradientes 
            # Primero aplicamos PPO, para que las redes aprendan a afinar sus movimientos
            # basándose en las ventajas calculadas por el Crítico.
            if self.use_ppo: # Interruptor para la ablación 
                for i, actor in enumerate(self.swarm):
                    states, actions, old_log_probs, rewards, dones = trajectories[i]

                    # Calculamos el GAE usando el Crítico centralizado
                    advantages, returns = self.compute_gae(states, rewards, dones) 

                    # Actualizamos los pesos del Actor actual y del Crítico compartido
                    self.ppo_update(actor, actor_optimizers[i], critic_optimizer, 
                                    states, actions, old_log_probs, advantages, returns, clip_eps=clip_eps)


            # Exploración global: PSO - ejambre
            # Despúes del gradiente aplicamos PSO. Esto empuja los pesos recién 
            # actualizados hacia las zonas de mayor recompensa histórica, introduciendo
            # diversidad y evitando que PPO se estanque en óptimos locales
            mean_velocity_magnitude = 0.0
            if self.use_pso:
                start_pso = time.time()
                for actor in self.swarm:
                    self.update_particle_pso(actor, w=w, c1=c1, c2=c2) # Pasamos w para el barrido
                    
                    # Calculamos la velocidad media de los pesos de la partícula
                    vel_mag = sum(torch.mean(torch.abs(v)).item() for v in actor.velocity)
                    mean_velocity_magnitude += vel_mag / len(actor.velocity)
                    
                time_spent_pso += (time.time() - start_pso)
                mean_velocity_magnitude /= self.num_particles
            
            # Guardamos todos los datos para las estadíticas
            logs.append({
                "iteration": it,
                "timesteps": total_timesteps,
                "mean_population_return": mean_population_fitness,
                "gbest_fitness": self.gbest_fitness,
                "mean_velocity": mean_velocity_magnitude, 
                "time_pso_cumulative": time_spent_pso,    
                "time_total_cumulative": time.time() - start_time_total
            })

        for env in envs: env.close()
        return pd.DataFrame(logs)



# Ejecución de los experimentos
if __name__ == "__main__":

    # Creamos la carpeta donde se guardará toda la telemtría 
    os.makedirs("logs", exist_ok=True)

    # Usamos 5 semillas para garantizar que los resultados no son casualidad
    seeds = [42, 101, 202, 303, 404] 
    num_iterations = 100 
    
    # Estudio de ablación
    # Comparamos el sistema completo frente a sus componentes aislados
    print("--- INICIANDO ESTUDIO DE ABLACIÓN ---")
    for seed in seeds:
        print(f"Entrenando Semilla {seed}...")
        
        # (a) Hibrído completo: PSO + PPO Completo (Enjambre n=4)
        agent_hybrid = PSO_PPO_Hybrid(num_particles=4, use_pso=True, use_ppo=True)
        df_hybrid = agent_hybrid.train(iterations=num_iterations, seed=seed)
        df_hybrid.to_csv(f"logs/ablation_a_hybrid_seed_{seed}.csv", index=False)
        
        # (b) Solo PPO (Política única n=1, apagamos el ejambre y PSO)
        # Algortimo RL estándar
        agent_ppo = PSO_PPO_Hybrid(num_particles=1, use_pso=False, use_ppo=True)
        df_ppo = agent_ppo.train(iterations=num_iterations, seed=seed)
        df_ppo.to_csv(f"logs/ablation_b_ppo_seed_{seed}.csv", index=False)

        # (c) Solo PSO (Enjambre n=4, sin gradientes PPO)
        # Las partículas aprender solo moviéndose por inercia y fitness
        agent_pso = PSO_PPO_Hybrid(num_particles=4, use_pso=True, use_ppo=False)
        df_pso = agent_pso.train(iterations=num_iterations, seed=seed)
        df_pso.to_csv(f"logs/ablation_c_pso_seed_{seed}.csv", index=False)


    # Barrido de hiperparámetros 
    # Reducimos un poco las iteraciones y usamos solo 3 semillas en los barridos
    # para cumplir con el requisito de máximo 2 horas de ejecución
    print("\n--- INICIANDO BARRIDO DE PARÁMETROS ---")
    sweep_iters = 60 
    
    # Barrido Estructural: Tamaño de población (n)
    # Evaluamos si tener un enjambre más grande compensa el mayor coste computacional.
    for n in [4, 8, 16, 32]:
        for seed in seeds[:3]: # Usamos 3 semillas 
            agent = PSO_PPO_Hybrid(num_particles=n, use_pso=True, use_ppo=True)
            df = agent.train(iterations=sweep_iters, seed=seed)
            df.to_csv(f"logs/sweep_n_{n}_seed_{seed}.csv", index=False)

    # Barrido Bioinspirado: Peso de inercia (w)
    # Evaluamos cuánta "inercia" histórica debe conservar una partícula.
    for w in [0.4, 0.7, 0.9]:
        for seed in seeds[:3]:
            agent = PSO_PPO_Hybrid(num_particles=4, use_pso=True, use_ppo=True)
            df = agent.train(iterations=sweep_iters, seed=seed, w=w)
            df.to_csv(f"logs/sweep_w_{w}_seed_{seed}.csv", index=False)

    # Barrido Bioinspirado: Coeficientes Cognitivo y Social (c1, c2)
    # Evaluamos el equilibrio entre hacer caso a la memoria propia (c1) vs la del líder (c2).
    # Los igualamos para simplificar el espacio de búsqueda.
    for c in [0.5, 1.0, 2.0]:
        for seed in seeds[:3]:
            agent = PSO_PPO_Hybrid(num_particles=4, use_pso=True, use_ppo=True)
            df = agent.train(iterations=sweep_iters, seed=seed, c1=c, c2=c)
            df.to_csv(f"logs/sweep_c_{c}_seed_{seed}.csv", index=False)

    print("¡Todos los experimentos completados y guardados en 'logs/'!")