from collections import deque

class MCTSMemory:
    
	def __init__(self, MEMORY_SIZE):
		self.MEMORY_SIZE = MEMORY_SIZE
		self.ltmemory = deque(maxlen=MEMORY_SIZE)
		self.stmemory = deque(maxlen=MEMORY_SIZE)

	def commit_stmemory(self, state, action_values):
		self.stmemory.append({
			'state': state,
			'name': str(state.flatten()),
			'action_values': action_values
			})

	def commit_ltmemory(self):
		for i in self.stmemory:
			self.ltmemory.append(i)
		self.clear_stmemory()

	def clear_stmemory(self):
		self.stmemory = deque(maxlen=self.MEMORY_SIZE)