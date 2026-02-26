class OrderStateMachine:
    def __init__(self):
        self.states = {
            "spring_decompression": self.spring_decompression_state,
            "aloft": self.aloft_state,
            "spring_compression": self.spring_compression_state
        }
        self.current_state = "aloft"

    def transition(self, event):
        if event in ["y_0 > 0", "y_0 <= 0", "X >= X_d"]:
            self.current_state = self.states[self.current_state](event)
        else:
            print(f"Invalid event: {event}")

    def spring_decompression_state(self, event):
        if event == "y_0 > 0":
            return "aloft"
        return self.current_state

    def aloft_state(self, event):
        if event == "y_0 <= 0":
            return "spring_compression"
        return self.current_state

    def spring_compression_state(self, event):
        if event == "X >= X_d":
            return "spring_decompression"
        return self.current_state

