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
		print("Player:", player)
		msg += pack_car(player)
		
	# Send ball
	msg += pack_physobj(j['ball'])

	print("sending to socket")
	sock.sendto(msg, (UDP_IP, UDP_PORT))

def render_state(state_json_str):
	j = json.loads(state_json_str)
	try:
		send_data_to_rsvis(j)
	except Exception:
		print("Exception while sending data:", traceback.format_except())