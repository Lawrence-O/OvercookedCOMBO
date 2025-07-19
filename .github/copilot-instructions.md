# Copilot AI Instructions for OvercookedCOMBO

Welcome to the OvercookedCOMBO repository. This guide highlights architecture, conventions, and workflows so Copilot (or other AI agents) can be productive immediately.

## 1. Project Layout & Big Picture
- `overcooked/`: Core Python package, split into:
  - `dataset/`: `OvercookedSequenceDataset` (HDF5-based) and `SingleEpisodeOvercookedDataset` for slicing episodes.
  - `diffusion/`: diffusion models (`goal_diffusion.py`), UNet implementations (`unet.py`), and `experiments_classes.py` (concept learning & evaluation).
  - `agent/`: `DiffusionPlannerAgent` wraps diffusion inference into action sequences.
  - `trainers/`: scripts for training models (`train_overcooked.py`, etc.).
  - `model_tests/`: evaluation and testing wrappers (e.g., `runner.py`, `test_classes.py`).
  - `utils/`: environment helpers (`managed_environment`), data transforms (`normalize_obs`), rendering (`overcooked_sample_renderer.py`).
- Legacy code under `AVDC/flowdiffusion/` mirrors early experiments; avoid modifying unless migrating functionality into `overcooked/`.
- External submodule `mapbt_package/` provides Overcooked MDP environment and policy pools via `mapbt_package.mapbt.envs.overcooked`.

## 2. Setup & Dependencies
- Environment files: `environment.yml` & `combo_env.yaml` configure conda; `requirements.txt` for pip installs.
- Primary Python dependencies: `torch`, `einops`, `moviepy`, `pygame`, `h5py`, `matplotlib`, and `mapbt_package` (included as subfolder).
- To start: `conda env create -f environment.yml` then `pip install -r requirements.txt` (or use `combo_env.yaml`).

## 3. Conventions & Patterns
- **Imports**: Always use the `overcooked.*` namespace after refactoring; avoid raw `sys.path` hacks.
- **Context managers** in `overcooked.utils.utils`:
  - `managed_environment(args, policy_name, n_envs)`: wraps env setup/reset/cleanup.
  - `managed_model_loading(runner, model_path)`: handles diffusion model load/unload and GPU cache.
- **Normalization**: `normalize_obs` / `convert_to_binary_obs` enforce channel-wise ranges ([-1,1] for continuous, binary channels).
- **Rendering**: `OvercookedSampleRenderer` in `overcooked/utils/overcooked_sample_renderer.py` for visualizing trajectories.
- **Config Logging**: training scripts use `config_to_log` & `config_to_wandb` to record hyperparameters;
  look at `overcooked/trainers/train_overcooked.py` for examples.

## 4. Key Workflows
- **Training**: Launch with `train_overcook_slurm.slurm` for cluster, or locally with:
  ```bash
  python -m overcooked.trainers.train_overcooked --horizon 8 --train_batch_size 32 ...
  ```
- **Evaluation**: Use `overcooked/model_tests/runner.py`:
  ```bash
  python -m overcooked.model_tests.runner --action_proposal_model_path <path> --idm_path <path> ...
  ```
- **SLURM**: `train_overcook_slurm.slurm` contains `sbatch` directives for GPUs and logs to `wandb` if configured.

## 5. Cross-Component Notes
- **Action Proposal vs. World Model**:
  - `UnetOvercookedActionProposal` in `overcooked/diffusion/unet.py` produces action distributions (1D UNet).
  - `GoalGaussianDiffusion` in `overcooked/diffusion/goal_diffusion.py` extends to image-conditioned diffusion.
- **Partners & IDMs**: inverse dynamics model loader at `agent/_load_idm_model()` uses `idm.inverse_dynamics.InverseDynamicsModel`.
- **Population Policies**: loaded via `Policy` from `mapbt_package.mapbt.algorithms.population.policy_pool`.

---

Questions or missing sections? Please provide feedback on unclear areas or suggest additions.
