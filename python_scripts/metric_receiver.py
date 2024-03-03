import wandb
import sys
import json

wandb_run = None

def init(py_exec_path, project, group, name):
	global wandb_run
	
	# Fix the path of our interpreter so wandb doesn't run RLGym_PPO instead of Python
	# Very strange fix for a very strange problem
	sys.executable = py_exec_path
	
	wandb_run = wandb.init(project = project, group = group, name = name)

def add_metrics(metrics):
	global wandb_run
	wandb_run.log(metrics)