try:
	import antipackage as apkg
except:
	print('install antipackage from: https://github.com/rmorshea/antipackage')

from github.rmorshea.misc import searchscript as ss
import re


def setter_handle(pattern, line, name):
	pre = pattern.sub(r'\1', line)
	post = pattern.sub(r'\2', line)
	return pre+'.'+name+' = '+post[1:-1]+'\n'

def underscore_handle(pattern, line, name):
	pre = pattern.sub(r'\1', line)
	post = pattern.sub(r'\2', line)
	if post.startswith(' = '):
		post = ','+post[3:-1]+')'
	else:
		post = ')'+post
	return pre[:-1]+".private('"+name+"'"+post


class MplReplacementLibrary(object):

	def __init__(self):
		self.working_name = None

	def setter(self, tool):
		name = self.working_name
		pattern = r'(.*)\.set_'+name+r'[^\(]*(\(.*\))[^\)]*'

		def handle(p, l):
			return setter_handle(p,l,name)

		args = (tool.rootdir,'py',pattern,None,handle)
		return ss.SearchReplace(*args, context=tool.context)

	def getter(self, tool):
		name = self.working_name

		repl_str = r'\1.'+name+r'\2'
		pattern = r'(.*)\.get_'+name+'\(\)(.*)'

		args = (tool.rootdir,'py',pattern,repl_str)
		return ss.SearchReplace(*args, context=tool.context)

	def underscore(self, tool):
		name = self.working_name
		pattern = r'(.*)\._'+name+r'(.*)'

		def handle(p, l):
			return underscore_handle(p,l,name)

		args = (tool.rootdir,'py',pattern,None,handle)
		return ss.SearchReplace(*args, context=tool.context)

	def __getattr__(self, key):
		if key.startswith('_'):
			self.working_name = key[1:]
			return self.underscore
		elif key.startswith('set_'):
			self.working_name = key[4:]
			return self.setter
		elif key.startswith('get_'):
			self.working_name = key[4:]
			return self.getter
		else:
			raise ValueError('the given key was not understood')


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
	rootdir = '/Users/RyanMorshead/Coding/GitHub/matplotlib/lib'