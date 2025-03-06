class DataProcessor:
    #Observe stream and add to list
    def __init__(self):
        self.data = []

    def process_data(self, message):
        """Handle incoming WebSocket messages."""
        self.data.append(message)
        print(f"Processed Data!")
    
    def working_data(self):
        return self.data

