try:
	import antipackage as apkg
except:
	print('install antipackage from: https://github.com/rmorshea/antipackage')

from github.rmorshea.misc import searchscript as ss
import re
import types


def setter_handle(pattern, line, name):
	pre, post = pattern.match(line).groups()
	return pre+'.'+name+' = '+post[1:-1]+'\n'

def underscore_handle(pattern, line, name):
	pre, post = pattern.match(line).groups()
	if post.startswith(' = '):
		post = ','+post[3:-1]+')'
	else:
		post = ') '+post
	return pre+".private('"+name+"'"+post


class MplReplacementLibrary(object):

	def __init__(self):
		self.working_name = None

	def setter(self, tool):
		name = self.working_name
		pattern = r'(.*)\.set_'+name+r'[\(]*(\(.*\))[^\)]*'

		def handle(p, l):
			return tool.handle_wrapper(setter_handle,p,l,name)

		args = (tool.rootdir,'py',pattern,None,handle)
		return ss.SearchReplace(*args, context=tool.context)

	def getter(self, tool):
		name = self.working_name
		pattern = r'(.*)\.get_'+name+'\(\)(.*)'

		def handle(p, l):
			method = lambda p,l,name: p.sub(r'\1.'+name+r'\2', l)
			return tool.handle_wrapper(method,p,l,name)

		args = (tool.rootdir,'py',pattern, None, handle)
		return ss.SearchReplace(*args, context=tool.context)

	def underscore(self, tool):
		name = self.working_name
		pattern = r'(.*)\._'+name+r'([^a-zA-Z0-9_](?:.*))'

		def handle(p, l):
			return tool.handle_wrapper(underscore_handle,p,l,name)

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

	def __init__(self, name, wrapper=None, context=0):
		self.context = context
		if wrapper:
			self.handle_wrapper = wrapper
		if self.lib is None:
			raise ValueError('no replacement library found')
		self._repl = getattr(self.lib, name)(self)

	def handle_wrapper(self, method, pattern, line, name):
		return method(pattern, line, name)

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