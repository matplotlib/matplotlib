try:
	import antipackage as apkg
except:
	print('install antipackage from: https://github.com/rmorshea/antipackage')

from github.rmorshea.misc import searchscript as ss
import re

class MplReplacementLibrary(object):

	@staticmethod
	def set_transform(tool):
		pattern = r'(.*)\.set_transform[^\(]*(\(.*\))[^\)]*'
		def handle(pattern, line):
			pre = pattern.sub(r'\1', line)
			post = pattern.sub(r'\2', line)
			return pre+'.transform = '+post[1:-1]+'\n'
		args = (tool.rootdir,'py',pattern,None,handle)
		return ss.SearchReplace(*args, context=tool.context)

	@staticmethod
	def get_transform(tool):
		pattern = r'(.*)\.get_transform\(\)(.*)'
		repl_str = r'\1.transform\2'
		args = (tool.rootdir,'py',pattern,repl_str)
		return ss.SearchReplace(*args, context=tool.context)

	@staticmethod
	def _transform(tool):
		pattern = r'(.*)\._transform(.*)'
		repl_str = r'\1.transform\2'
		args = (tool.rootdir,'py',pattern,repl_str)
		return ss.SearchReplace(*args, context=tool.context)

	@staticmethod
	def _stale(tool):
		pattern = r'(.*)\._stale(.*)'
		repl_str = r'\1.stale\2'
		args = (tool.rootdir,'py',pattern,repl_str)
		return ss.SearchReplace(*args, context=tool.context)

class ReplaceTool(object):

	lib = None
	rootdir = None

	def __init__(self, name, context=0):
		self.context = context
		if self.lib is None:
			raise ValueError('no replacement library found')
		self._repl = getattr(self.lib, name)(self)

	def find_replacements(self):
		self._repl.find_replacements()

	def review_replacements(self):
		self._repl.review_replacements()

	def perform_replacements(self):
		self._repl.perform_replacements()

	def refresh(self):
		self._repl.refresh()

	def help():
		self._repl.help()

	def undo(self):
		self._repl.undo()

class MatplotlibReplace(ReplaceTool):

	lib = MplReplacementLibrary()
	rootdir = '/Users/RyanMorshead/Coding/GitHub/matplotlib/lib/matplotlib/'	