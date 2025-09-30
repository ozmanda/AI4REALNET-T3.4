from typing import Dict


class NegotiationProxy:
    def __init__(self):
        self.params: Dict = {}

    def init_params(self, negotiation_params: Dict):
        self.params = negotiation_params

    def adjust_param(self, new_params: Dict):
        """
        Adjust of negotiation parameters during runtime based on human input.
        """
        for key, value in new_params.items():
            if key in self.params:
                self.params[key] = value

    def perform_negotiation(self):
        """
        Initialize the negotiation process with the current parameters.
        """
        pass
