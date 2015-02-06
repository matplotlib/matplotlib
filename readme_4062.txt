The objective of this Pull Request is to make installation more robust on mac's regarding freetype paths   

Analysis on existing commit(master 8f10470):
+ Setup.py has a list of pre-requisit packages "mpl_packages."
+ A for loop after "__main__" check goes through this list.
+ If package.check() returns a value other than None, setup recognizes that package.

+ Setupext.py has a class for Freetype incluidng check().
+ FreeType check() method searches for a header file name "ft2build.h."
