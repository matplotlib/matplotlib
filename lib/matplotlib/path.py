"""
Data structure to store paths.  The basic structure is a sequence of
tuples.  The first element of the tuple is an integer representing the
path element type code, eg MOVETO, LINETO.  The remaining elements of
the tuple are the paremters for that element, and may vary in length

Eg, MOVETO and LINETO take 2 params (x and y) and CURVE3 takes 4 (the
x and y control points and the x and y to points)

  MOVETO, x, y
  LINETO, x, y
  CURVE3, xctrl, yctrl, xto, yto
  CURVE3, xctrl1, yctrl1, xctrl2, yctrl2, xto, yto
  ENDPOLY, 0  # close poly, don't fill
  ENDPOLY, 1, R, G, B, A  # close poly, fill with rgba

Where MOVETO, etc are module integer constants  
  STOP    = 0 
  MOVETO  = 1
  LINETO  = 2
  CURVE3  = 3 
  CURVE4  = 4 
  ENDPOLY = 6


"""

STOP    = 0 
MOVETO  = 1
LINETO  = 2
CURVE3  = 3 
CURVE4  = 4 
ENDPOLY = 6
