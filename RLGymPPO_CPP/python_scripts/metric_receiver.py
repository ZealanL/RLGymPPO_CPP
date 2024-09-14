import signal
import site
import sys
import os

wandb_run = None

# Takes in the python executable path, the three wandb init strings, and optionally the current run ID
# Returns the ID of the run (either newly created or resumed)
def init(py_exec_path, project, group, name, id = None):
	"""Takes in the python executable path, the three wandb init strings, and optionally the current run ID. Returns the ID of the run (either newly created or resumed)

	Args:
		py_exec_path (str): Python executable path, necessary to fix a bug where the wrong interpreter is used
		project (str): Wandb project name
		group (str): Wandb group name
		name (str): Wandb run name
		id (str, optional): Id of the wandb run, if None, a new run is created

	Raises:
		Exception: Failed to import wandb

	Returns:
		str: The id of the created or continued run
	"""

	global wandb_run
	
	# Fix the path of our interpreter so wandb doesn't run RLGym_PPO instead of Python
	# Very strange fix for a very strange problem
	sys.executable = py_exec_path
	
	try:
		site_packages_dir = os.path.join(os.path.join(os.path.dirname(py_exec_path), "Lib"), "site-packages")
		sys.path.append(site_packages_dir)
		site.addsitedir(site_packages_dir)
		import wandb
	except Exception as e:
		raise Exception(f"""
			FAILED to import wandb! Make sure RLGymPPO_CPP isn't using the wrong Python installation.
			This installation's site packages: {site.getsitepackages()}
			Exception: {repr(e)}"""
		)
	
	print("Calling wandb.init()...")
	if not (id is None) and len(id) > 0:
		wandb_run = wandb.init(project = project, group = group, name = name, id = id, resume = "allow")
	else:
		wandb_run = wandb.init(project = project, group = group, name = name)
	return wandb_run.id

def add_metrics(metrics):
	"""Logs metrics to the wandb run

	Args:
		metrics (Dict[str, Any]): The metrics to log
	"""
	global wandb_run
	wandb_run.log(metrics)


def end(_signal):
	"""Runs post-mortem tasks

	Args:
		signal (int): Received signal
	"""
	print(f"Received signal {_signal} ({signal.strsignal(_signal)})")

	# SIGBREAK crashes wandb_run.finish on a WinError[10054].

	if _signal != signal.Signals.SIGBREAK.value:
		wandb_run.finish()