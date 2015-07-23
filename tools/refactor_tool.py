import antipackage as apkg
from github.rmorshea.searchscript import searchscript as ss
import re

class MatplotlibReplace(object):

	rootdir = '/Users/RyanMorshead/Coding/GitHub/matplotlib/lib/matplotlib/'

	def __init__(self, context):
		self.context = context

	def repl_set_transform(self):
		pattern = r'(.*)\.set_transform[^\(]*(\(.*\))[^\)]*'
		def handle(pattern, line):
			pre = pattern.sub(r'\1', line)
			post = pattern.sub(r'\2', line)
			return pre+'.transform = '+post[1:-1]+'\n'
		args = (self.rootdir,'py',pattern,None,handle)
		return ss.SearchReplace(*args, context=self.context)

	def repl_get_transform(self):
		pattern = r'(.*)\.get_transform\(\)(.*)'
		repl_str = r'\1.transform\2'
		args = (self.rootdir,'py',pattern,repl_str)
		return ss.SearchReplace(*args, context=self.context)

	def repl_transform(self):
		pattern = r'(.*)\._transform(.*)'
		repl_str = r'\1.transform\2'
		args = (self.rootdir,'py',pattern,repl_str)
		return ss.SearchReplace(*args, context=self.context)

	def repl_stale(self):
		pattern = r'(.*)\._stale(.*)'
		repl_str = r'\1.stale\2'
		args = (self.rootdir,'py',pattern,repl_str)
		return ss.SearchReplace(*args, context=self.context)