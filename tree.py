import math
from copy import deepcopy


class Tree:
	def __init__(self, alphabet, probability=None, addr='lambda'):
		self.alphabet = alphabet
		if probability == None:
			self.children = [None] * alphabet
			self.leaf = False
		else:
			# verify len(probability)==alphabet
			self.verify_leaf(probability)
			self.leaf = True


	def __getitem__(self, key):
		key_child = self.children[int(key[0])]
		if len(key) == 1 or key_child.leaf:
			return key_child
		return key_child.__getitem__(key[1:])

	def __setitem__(self, key, value):
		'''
		code for adding a leaf. value should be a list of probabilities
		'''
		if key == '':
			self.verify_leaf(value)
			self.leaf = True

		list_key = int(key[0])
		if len(key)==1:
			self.children[int(key)] = Tree(self.alphabet, value)

		else:
			if self.children[list_key]==None:
				self.children[list_key] = Tree(self.alphabet)
			self.children[list_key][key[1:]] = value

	def verify(self):
		if self.leaf:
			return True
		for child in self.children:
			if not bool(child):
				return False
			if not child.verify():
				return False
		return True

	def verify_leaf(self, probability):
		cumul = 0
		for p in probability:
			cumul += p
			if p < 0:
				raise ValueError("Negative probability.")
		if not math.isclose(cumul, 1, rel_tol=1e-9):
			raise ValueError("Probability does not sum to 1.")
		self.probability = probability

	def __str__(self):
		if self.leaf:
			return str(self.probability)
		else: 
			return str(self.children)

	def p(self, margins=None):
		if margins == None:
			return self.probability

		if isinstance(margins, int):
			margins = (margins,)

		probability = deepcopy(self.probability)
		denominator = 1
		for margin in margins:
			denominator -= probability[margin]
			probability[margin] = 0
		return [prob/denominator for prob in probability]
	
	def draw(self, D, filename):
		links, nodes = self.linker('M', D)
		with open(filename, 'w') as f:
			for node in nodes:
				f.write(f"{{ data: {{ id: '{node}', label: '{node}' }} }},\n")

			for (source, target) in links:
				f.write(f"{{ data: {{ source: '{source}', target: '{target}' }} }},\n")


	def linker(self, addr, d):
		if d == 0 or self.leaf:
			return [], [addr]
		
		nodes = [addr]
		links = []
		for i, child in enumerate(self.children):
			child_addr = addr+str(i)
			links.append((addr, child_addr))
			desc_links, desc_nodes = child.linker(child_addr, d-1)
			links += desc_links
			nodes += desc_nodes

		return links, nodes
