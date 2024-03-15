import sys
import os
import struct
import numpy as np
from collections import OrderedDict, namedtuple

import torch

from rlgym_ppo.ppo import DiscreteFF, ValueEstimator

def model_info_from_dict(loaded_dict):
	state_dict = OrderedDict(loaded_dict)

	bias_counts = []
	weight_counts = []
	for key, value in state_dict.items():
		if ".weight" in key:
			weight_counts.append(value.numel())
		if ".bias" in key:
			bias_counts.append(value.size(0))
			
	inputs = int(weight_counts[0] / bias_counts[0])
	outputs = bias_counts[-1]
	layer_sizes = bias_counts[:-1]
	
	return inputs, outputs, layer_sizes

def rename_model_state_dict(state_dict):
	keys = []
	for key in state_dict.keys():
		keys.append(key)
	
	for key in keys:
		if not "model." in key:
			state_dict["model." + key] = state_dict[key]
			del state_dict[key]
			
	return state_dict

def read_optimizer_state(optim, path):
	fin = open(path, 'rb')
	prefix = struct.unpack('I', fin.read(4))[0]
	
	with torch.no_grad():
		groups = optim.param_groups
		group_amount = struct.unpack('I', fin.read(4))[0]
		if len(groups) != group_amount:
			raise Exception("Bad group count ({} != {})".format(len(groups), group_amount))
		for group in groups:
			params = group['params']
			param_amount = struct.unpack('I', fin.read(4))[0]
			if len(params) != param_amount:
				raise Exception("Bad param count ({} != {})".format(len(params), param_amount))
			for param in params:
				tensor_size = struct.unpack('Q', fin.read(8))[0]
				if param.numel() != tensor_size:
					raise Exception("Bad tensor size ({} != {})".format(param.numel(), tensor_size))
				
				new_vals = []
				for i in range(param.numel()):
					new_vals.append(struct.unpack('f', fin.read(4))[0])
				
				param.copy_(torch.tensor(new_vals).reshape_as(param))

def write_optimizer_state(optim, path):
	fout = open(path, 'wb')
	bytes = struct.pack('I', 0xB73AC038) # Write prefix	
	groups = optim.param_groups
	bytes += struct.pack('I', len(groups))
	for group in groups:
		params = group['params']
		bytes += struct.pack('I', len(params))
		for param in params:
			bytes += struct.pack('Q', param.numel())
			for val in param.detach().flatten().numpy():
				bytes += struct.pack('f', val)
				
	fout.write(bytes)
	
def get_optim_mean(optim):
	total_mean = 0
	for group in optim.param_groups:
		for param in group['params']:
			total_mean += abs(param.detach().mean().item())
	return total_mean
	
def make_models_from_dicts(policy_state_dict, critic_state_dict):
	policy_inputs, policy_outputs, policy_sizes = model_info_from_dict(policy_state_dict)
	critic_inputs, critic_outputs, critic_sizes = model_info_from_dict(critic_state_dict)
	
	device = torch.device("cpu")
	policy = DiscreteFF(policy_inputs, policy_outputs, policy_sizes, device)
	critic = ValueEstimator(critic_inputs, critic_sizes, device)
	return policy, critic
	
def main():

	if len(sys.argv) != 3:
		sys.exit("Invalid argument count, arguments should be \"<to_cpp/to_python> <checkpoint path>\"")

	to_arg = sys.argv[1]
	if to_arg == 'to_cpp':
		to_cpp = True
	elif to_arg == 'to_python':
		to_cpp = False
	else:
		sys.exit("Invalid arguments, please specify \"-save\" or \"-load\" for the first argument.")
		
	path = sys.argv[2]
	
	device = torch.device('cpu')
	
	if to_cpp:
		print("Loading state dicts...")
		policy_state_dict = torch.load(os.path.join(path, "PPO_POLICY.pt"))
		critic_state_dict = torch.load(os.path.join(path, "PPO_VALUE_NET.pt"))
		policy_optim_state_dict = torch.load(os.path.join(path, "PPO_POLICY_OPTIMIZER.pt"))
		critic_optim_state_dict = torch.load(os.path.join(path, "PPO_VALUE_NET_OPTIMIZER.pt"))
		
		print("Creating models...")
		policy, critic = make_models_from_dicts(policy_state_dict, critic_state_dict)
		
		policy_optim = torch.optim.Adam(policy.parameters())
		critic_optim = torch.optim.Adam(critic.parameters())
		
		print("Applying state dicts...")
		
		policy.load_state_dict(policy_state_dict)
		critic.load_state_dict(critic_state_dict)
		
		policy_optim.load_state_dict(policy_optim_state_dict)
		critic_optim.load_state_dict(critic_optim_state_dict)
		
		print("Saving for RLGymPPO_CPP...")
		policy_ts = torch.jit.script(policy.model)
		critic_ts = torch.jit.script(critic.model)
		
		output_path = "cpp_checkpoint"
		os.makedirs(output_path, exist_ok = True)
		torch.jit.save(policy_ts, output_path + "/PPO_POLICY.lt")
		torch.jit.save(critic_ts, output_path + "/PPO_CRITIC.lt")
		write_optimizer_state(policy_optim, output_path + "/PPO_POLICY_OPTIM.rlps")
		write_optimizer_state(critic_optim, output_path + "/PPO_CRITIC_OPTIM.rlps")
		
	else:
		print("Loading models...")
		policy = torch.jit.load(os.path.join(path, "PPO_POLICY.lt"))
		critic = torch.jit.load(os.path.join(path, "PPO_CRITIC.lt"))
		policy_inputs, policy_outputs, policy_sizes = model_info_from_dict(policy.state_dict())
		
		print("Creating optimizers...")
		policy_py, critic_py = make_models_from_dicts(policy.state_dict(), critic.state_dict())
		
		policy_optim = torch.optim.Adam(policy_py.parameters())
		critic_optim = torch.optim.Adam(critic_py.parameters())

		print("Populating optimizers...")
		action, log_prob = policy_py.get_action(np.zeros(policy_inputs))
		value = critic_py.forward(np.zeros(policy_inputs))
		log_prob.backward()
		value.backward()
		policy_optim.step()
		critic_optim.step()
		
		print("Loading optimizers...")
		read_optimizer_state(policy_optim, os.path.join(path, "PPO_POLICY_OPTIM.rlps"))
		read_optimizer_state(critic_optim, os.path.join(path, "PPO_CRITIC_OPTIM.rlps"))
		print("Saving for rlgym-ppo...")
		output_path = "python_checkpoint"
		os.makedirs(output_path, exist_ok = True)
		
		policy_state_dict = rename_model_state_dict(policy.state_dict())
		critic_state_dict = rename_model_state_dict(critic.state_dict())
		
		torch.save(policy_state_dict, output_path + "/PPO_POLICY.pt")
		torch.save(critic_state_dict, output_path + "/PPO_VALUE_NET.pt")
		torch.save(policy_optim.state_dict(), output_path + "/PPO_POLICY_OPTIMIZER.pt")
		torch.save(critic_optim.state_dict(), output_path + "/PPO_VALUE_NET_OPTIMIZER.pt")
		
	print(
		"Done! Partial " + ("RLGymPPO_CPP" if to_cpp else "rlgym-ppo") + 
		" checkpoint generated at \"" + output_path + "\"."
	)
	print("NOTE: State JSON not included (just make a new one and copy over the vars you want).")
	print("NOTE: Make sure the obs/actions/model sizes all match.")
	
if __name__=='__main__':
	main()