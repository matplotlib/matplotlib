from api import rcParams, mplConfig

print 'loaded your old rcParams["backend"]:', rcParams['backend']
print 'changing rcParams["backend"] to cairo'
rcParams["backend"] = 'cairo'
print 'mplConfig.backend.use is now :', mplConfig.backend.use
print 'changing rcParams["backend"] to BogusBackend:'
rcParams["backend"] = 'BogusBackend'
