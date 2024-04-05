import sys
import json
import traceback

import socket
import struct

# =======================
# Example implementation of render receiver, using RocketSimVis
# =======================

# Send to RocketSimVis
UDP_IP = "127.0.0.1"
UDP_PORT = 9273

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP

def pack_vec(vec):
	return struct.pack("<f", vec[0]) + struct.pack("<f", vec[1]) + struct.pack("<f", vec[2])

def pack_physobj(physobj):
	return (
		pack_vec(physobj['pos']) + 
		pack_vec(physobj['forward']) + pack_vec(physobj['right']) + pack_vec(physobj['up']) + 
		pack_vec(physobj['vel']) + 
		pack_vec(physobj['ang_vel'])
		)

def pack_car(player):
	bytes = b""
	
	# Team number
	bytes += struct.pack("B", int(player['team_num']))
	
	# Car physics state
	bytes += pack_physobj(player['phys'])
	
	# Car boost
	bytes += struct.pack("<f", player['boost_amount'])
	
	# Car demoed
	bytes += struct.pack("B", player['is_demoed'])
	
	# Car controls (not implemented yet)
	bytes += struct.pack("<f", 0)
	bytes += struct.pack("B", 0)
	bytes += struct.pack("B", 0)
	bytes += struct.pack("B", 0)
	
	return bytes

# RocketSimVis needs to know where the boost pads are
# This is the order of boost locations from RLGym
BOOST_LOCATIONS = ( 
	(0, -4240, 70), (-1792, -4184, 70), (1792, -4184, 70), (-3072, -4096, 73), (3072, -4096, 73), 
	(- 940, -3308, 70), (940, -3308, 70), (0, -2816, 70), (-3584, -2484, 70), (3584, -2484, 70), 
	(-1788, -2300, 70), (1788, -2300, 70), (-2048, -1036, 70), (0, -1024, 70), (2048, -1036, 70), 
	(-3584, 0, 73), (-1024, 0, 70), (1024, 0, 70), (3584, 0, 73), (-2048, 1036, 70), (0, 1024, 70), 
	(2048, 1036, 70), (-1788, 2300, 70), (1788, 2300, 70), (-3584, 2484, 70), (3584, 2484, 70), 
	(0, 2816, 70), (- 940, 3310, 70), (940, 3308, 70), (-3072, 4096, 73), (3072, 4096, 73), 
	(-1792, 4184, 70), (1792, 4184, 70), (0, 4240, 70)
)

def send_data_to_rsvis(j):
	msg = b""
	
	# Prefix signature
	# This is required at the start of all packets so that RocketSimVis knows its valid data
	msg += struct.pack("I", 0xA490E7B3)
	
	# Arena tick count (not implemented)
	msg += struct.pack("I", 0)
	
	# Send cars
	players = j['players']
	msg += struct.pack("I", len(players))
	for player in players:
		msg += pack_car(player)
		
	# Send boost pads
	pads = j['boost_pads']
	msg += struct.pack("I", len(pads))
	for i in range(len(pads)):
		pos = BOOST_LOCATIONS[i]
		is_active = pads[i]
		cooldown = 0 # Not implemented
		
		msg += pack_vec(pos)
		msg += struct.pack("B", is_active)
		msg += struct.pack("<f", cooldown)
	
	# Send ball
	msg += pack_physobj(j['ball'])

	sock.sendto(msg, (UDP_IP, UDP_PORT))

def render_state(state_json_str):
	j = json.loads(state_json_str)
	try:
		if 'state' in j:
			send_data_to_rsvis(j['state'])
		else:
			send_data_to_rsvis(j)
	except Exception as err:
		print("Exception while sending data:")
		traceback.print_exc()