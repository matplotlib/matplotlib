//----------------------------------*-C++-*----------------------------------//
// Copyright 1998 The Regents of the University of California. 
// All rights reserved. See Legal.htm for full text and disclaimer.
//---------------------------------------------------------------------------//

#include "CXX/Objects.hxx"
namespace Py {

Type Object::type () const
	{ 
	return Type (PyObject_Type (p), true);
	}

String Object::str () const
	{
	return String (PyObject_Str (p), true);
	}

String Object::repr () const
	{ 
	return String (PyObject_Repr (p), true);
	}

std::string Object::as_string() const
	{
	return static_cast<std::string>(str());
	}

bool Object::isType (const Type& t) const
	{ 
	return type ().ptr() == t.ptr();
	}

Char::operator String() const
	{
	return String(ptr());
	}

// TMM: non-member operaters for iterators - see above
// I've also made a bug fix in respect to the cxx code
// (dereffed the left.seq and right.seq comparison)
bool operator==(const Sequence::iterator& left, const Sequence::iterator& right)
	{
	return left.eql( right );
	}

bool operator!=(const Sequence::iterator& left, const Sequence::iterator& right)
	{
	return left.neq( right );
	}

bool operator< (const Sequence::iterator& left, const Sequence::iterator& right)
	{
	return left.lss( right );
	}

bool operator> (const Sequence::iterator& left, const Sequence::iterator& right)
	{
	return left.gtr( right );
	}

bool operator<=(const Sequence::iterator& left, const Sequence::iterator& right)
	{
	return left.leq( right );
	}

bool operator>=(const Sequence::iterator& left, const Sequence::iterator& right)
	{
	return left.geq( right );
	}

// now for const_iterator
bool operator==(const Sequence::const_iterator& left, const Sequence::const_iterator& right)
	{
	return left.eql( right );
	}

bool operator!=(const Sequence::const_iterator& left, const Sequence::const_iterator& right)
	{
	return left.neq( right );
	}

bool operator< (const Sequence::const_iterator& left, const Sequence::const_iterator& right)
	{
	return left.lss( right );
	}

bool operator> (const Sequence::const_iterator& left, const Sequence::const_iterator& right)
	{
	return left.gtr( right );
	}

bool operator<=(const Sequence::const_iterator& left, const Sequence::const_iterator& right)
	{
	return left.leq( right );
	}

bool operator>=(const Sequence::const_iterator& left, const Sequence::const_iterator& right)
	{
	return left.geq( right );
	}

// For mappings:
bool operator==(const Mapping::iterator& left, const Mapping::iterator& right)
	{
	return left.eql( right );
	}

bool operator!=(const Mapping::iterator& left, const Mapping::iterator& right)
	{
	return left.neq( right );
	}

// now for const_iterator
bool operator==(const Mapping::const_iterator& left, const Mapping::const_iterator& right)
	{
	return left.eql( right );
	}

bool operator!=(const Mapping::const_iterator& left, const Mapping::const_iterator& right)
	{
	return left.neq( right );
	}

// TMM: 31May'01 - Added the #ifndef so I can exlude iostreams.
#ifndef CXX_NO_IOSTREAMS
// output

std::ostream& operator<< (std::ostream& os, const Object& ob)
	{
	return (os << static_cast<std::string>(ob.str()));
	}  
#endif

} // Py
