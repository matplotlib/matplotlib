The objective of this Pull Request is to make installation more robust on mac's regarding freetype paths   

Analysis on existing commit(master 8f10470):
+ Setup.py has a list of pre-requisit packages "mpl_packages."
+ A for loop after "__main__" check goes through this list.
+ If package.check() returns a value other than None, setup recognizes that package.

+ Setupext.py has a class for Freetype incluidng check().
+ FreeType check() method searches for a header file name "ft2build.h."

+ On mac, FreeType check() method works as follows.
++ Using "freetype-config --ftversion" command, check version.
++ check() returns the return value from _check_for_pkg_config().

+++ _check_for_pkg_config() belongs to superclass SetupPackage of setupext.py
+++ _check_for_pkg_config() calls self.get_extension() to obtain ext object
		Here, because FreeType doesn't overload get_extension() method, 
			calls SetupPackage.get_extension() returning None
+++ Then, _check_for_pkg_config() calls make_extension('test', []) for ext object
			whose include_dirs covers
				['/usr/local/include', '/usr/include', '/usr/X11/include', '/opt/X11/include', '.']
+++ _check_for_pkg_config() returns result from check_include_file() 

++++ check_include_file() searches for include files within paths in ext.include_dirs 
