class Buffer:
    def __init__(self, buffer_size):
        self.transitions = []
        self.buffer_size = buffer_size

    def store(self, transition):
        self.transitions.append(transition)
        if len(self.transitions) > self.buffer_size:
            self.transitions = self.transitions[-buffer_size:]
