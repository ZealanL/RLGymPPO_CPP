import socket
import struct
import json

###############################################
# Your code here:

import wandb
import sys

# Initialize wandb from launch arguments
if len(sys.argv) != 4:
	sys.exit('[MR] FATAL ERROR: Bad launch arguments, they should be "metrics_receiver.py <group> <project> <run>"')
WANDB_RUN = wandb.init(project = sys.argv[1], group = sys.argv[2], name = sys.argv[3])

class MetricsHandler:
	def __init__(self):
		pass

	def process_json(self, j):
		WANDB_RUN.log(j['metrics'])


################################################
# Backend networking code:

LOG_VERBOSE = False

UDP_IP = "127.0.0.1"
UDP_PORT_CPP = 3942
UDP_PORT_PY = 3943
CONNECT_CODE = 0x1AB80D60
COMM_PREFIX = 0x1AB80D61
ACK_PREFIX = 0x1AB80D62

def send_connect_code(sock):
	bytes = struct.pack("I", 0x1AB80D60)
	sock.sendto(bytes, (UDP_IP, UDP_PORT_CPP))

def main():
	print("[MR] Starting...")
	
	handler = MetricsHandler()
	
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP
	sock.bind((UDP_IP, UDP_PORT_PY))
	
	print("[MR] Bound...")
	
	send_connect_code(sock)
	
	print("[MR] Connection code sent.")
	
	while True:
		bytes, addr = sock.recvfrom(1000 * 1000)
		
		assert len(bytes) >= 12 # Smaller than this is not valid

		prefix = struct.unpack('I', bytes[0:4])[0]
		assert prefix == COMM_PREFIX
		
		if LOG_VERBOSE:
			print("[MR] Recieved")
		msg_id = struct.unpack('I', bytes[4:8])
		
		if LOG_VERBOSE:
			print("[MR] Message id:", msg_id)
		
		json_len = struct.unpack('I', bytes[8:12])
		json_str = str(bytes[12:], encoding="ascii")
		
		if LOG_VERBOSE:
			print("[MR] JSON Str:", json_str)
		
		j = json.loads(json_str)
		handler.process_json(j)
		

if __name__=='__main__':
	main()
