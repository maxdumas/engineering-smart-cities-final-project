from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()
tf1.enable_eager_execution()

from ray.rllib.algorithms.ppo import PPO
import ray
from tqdm import tqdm

from esc.epanet_env import N_SIMULATION_STEPS, EPANETEnv

N_TRAINING_ITERATIONS = 1000
gpu_count = 1.0
driver_gpu = 1.0
num_workers = 9
num_gpus_per_worker = (gpu_count - driver_gpu) / num_workers
# Configure the algorithm.
config = {
    "env_config": {},
    # Use 5 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": num_workers,
    "create_env_on_driver": False,
    "num_gpus": driver_gpu,
    "num_gpus_per_worker": num_gpus_per_worker,
    "horizon": N_SIMULATION_STEPS,
    "train_batch_size": N_SIMULATION_STEPS * num_workers,
    "rollout_fragment_length": N_SIMULATION_STEPS,
    "batch_mode": "complete_episodes",
    "lr": 0.005,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "tf2",
    "eager_tracing": True,
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
        "use_lstm": True,
    },
    "recreate_failed_workers": True,
    "log_sys_usage": False,
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    # "evaluation_config": {
    #     "render_env": True,
    # },
    "log_level": "ERROR",
}

ray.init(
    num_cpus=10, num_gpus=1, runtime_env={"env_vars": {"CUDA_VISIBLE_DEVICES": ""}}
)

# Create our RLlib Trainer.
algo = PPO(env=EPANETEnv, config=config)
# algo.restore(
#     "/Users/maxdumas/ray_results/PPO_EPANETEnv_2022-12-04_17-41-00x8n0zze7/checkpoint_000012"
# )

# Print the final config
print(algo.config)

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for i in tqdm(range(N_TRAINING_ITERATIONS)):
    print(algo.train())

    if i % 5 == 0:
        checkpoint = algo.save()
        print(f"Checkpoint saved to {checkpoint}")

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
algo.evaluate()
