import platform
import os
import socket
import time

import psutil

from rlbot.agents.base_independent_agent import BaseIndependentAgent
from rlbot.botmanager.helper_process_request import HelperProcessRequest
from rlbot.utils.structures import game_interface
from rlbot.agents.base_agent import BOT_CONFIG_AGENT_HEADER
from rlbot.parsing.custom_config import ConfigHeader, ConfigObject

class BaseCPPAgent(BaseIndependentAgent):

    def __init__(self, name, team, index):
    
        print("Creating RLGymPPO_CPP python agent...")
    
        super().__init__(name, team, index)
        self.port = self.read_port_from_file()
        self.is_retired = False
        self.cpp_executable_path = None
        
    def run_independently(self, terminate_request_event):
        while not terminate_request_event.is_set():
            message = f"add\n{self.name}\n{self.team}\n{self.index}\n{game_interface.get_dll_directory()}"
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect(("127.0.0.1", self.port))
                s.send(bytes(message, "ASCII"))
                s.close()
            except ConnectionRefusedError:
                self.logger.warn("Could not connect to server!")

            time.sleep(1)
        else:
            self.retire()

    def get_helper_process_request(self):
        if self.is_executable_configured():
            return HelperProcessRequest(python_file_path=None, key=__file__ + str(self.port), executable=self.cpp_executable_path, exe_args = ["-dll-path", game_interface.get_dll_directory()])
        return None

    def is_executable_configured(self):
        return self.cpp_executable_path is not None and os.path.isfile(self.cpp_executable_path)

    def get_extra_pids(self):
        """
        Gets the list of process ids that should be marked as high priority.
        :return: A list of process ids that are used by this bot in addition to the ones inside the python process.
        """
        while not self.is_retired:
            if platform.system() == 'Linux':
                return []
            for proc in psutil.process_iter():
                for conn in proc.connections():
                    if conn.laddr.port == self.port:
                        self.logger.debug(f"C++ socket server for {self.name} appears to have pid {proc.pid}")
                        return [proc.pid]
            if self.is_executable_configured():
                return []
            time.sleep(1)
            if self.cpp_executable_path is None:
                self.logger.info("Can't auto-start C++ executable because no executable is configured. "
                                 "Please start the C++ bot manually!")
            else:
                self.logger.info(f"Can't auto-start C++ executable because {self.cpp_executable_path} "
                                 "is not found. Please start the C++ bot manually!")

    def retire(self):
        port = self.read_port_from_file()
        message = f"remove\n{self.index}"

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", port))
            s.send(bytes(message, "ASCII"))
            s.close()
        except ConnectionRefusedError:
            self.logger.warn("Could not connect to server!")
        self.is_retired = True

    def read_port_from_file(self):
        try:
            location = self.get_port_file_path()

            with open(location, "r") as port_file:
                return int(port_file.readline().rstrip())

        except ValueError:
            self.logger.warn("Failed to parse port file!")
            raise

    def get_port_file_path(self):
        # Look for a port.cfg file in the same directory as THIS python file.
        return os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), 'port.cfg'))

    def load_config(self, config_header: ConfigHeader):
        self.cpp_executable_path = config_header.getpath('cpp_executable_path')
        self.logger.info("C++ executable is configured as {}".format(self.cpp_executable_path))

    @staticmethod
    def create_agent_configurations(config: ConfigObject):
        params = config.get_header(BOT_CONFIG_AGENT_HEADER)
        params.add_value('cpp_executable_path', str, default=None, description='Relative path to the executable that runs the cpp bot.')
