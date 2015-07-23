import antipackage as apkg
from github.rmorshea.searchscript import searchscript as ss

class MatplotlibReplace(object):

	rootdir = '/Users/RyanMorshead/Coding/GitHub/matplotlib/lib/matplotlib/'

	def __init__(self, context):
		self.context = context

	def repl_set_transform(self):
		pattern = r'(.*)\.set_transform[^\(]*(\(.*\))[^\)]*'
		def handle(patter, line):
		    pre = sr.pattern.sub(r'\1', line)
		    post = sr.pattern.sub(r'\2', line)
		    return pre+'.transform = '+post[1:-1]+'\n'
		args = (self.rootdir,'py',pattern,None,handle)
	    return ss.SearchReplace(*args, context=self.context)

	def repl_transform(self):
		pattern = r'(.*)\._transform = (.*)'
		repl_str = '\1.transform = \2'
		args = (self.rootdir,'py',pattern,repl_str)
		return ss.SearchReplace(*args, context=self.context)