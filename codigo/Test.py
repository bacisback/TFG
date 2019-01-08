from abc import abstractmethod, ABCMeta
class Test:
	"""
	Clase abstracta para referirnos a los test 
	de independencia que implementaremos
	"""

	__metaclass__ = ABCMeta

	def __init__(self,nombre):
		self.nombre = nombre

	@abstractmethod
	def test(x,y):
		pass
	@abstractmethod
	def generar_distribucion_empirica(x,y):
		pass
	def test_tiempos(n):
		pass
