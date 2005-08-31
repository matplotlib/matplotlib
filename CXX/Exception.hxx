//----------------------------------*-C++-*----------------------------------//
// Copyright 1998 The Regents of the University of California. 
// All rights reserved. See LEGAL.LLNL for full text and disclaimer.
//---------------------------------------------------------------------------//

#ifndef __CXX_Exception_h
#define __CXX_Exception_h

#include "Python.h"
#include "CXX/Version.hxx"
#include "CXX/Config.hxx"
#include "CXX/IndirectPythonInterface.hxx"

#include <string>
#include <iostream>

// This mimics the Python structure, in order to minimize confusion
namespace Py
	{
	class ExtensionExceptionType;

	class Object;

	class Exception
		{
	public:
		Exception( ExtensionExceptionType &exception, const std::string& reason );
		Exception( ExtensionExceptionType &exception, Object &reason );

		explicit Exception ()
			{}
		
		Exception (const std::string& reason)
			{
			PyErr_SetString (Py::_Exc_RuntimeError(), reason.c_str());
			}
		
		Exception (PyObject* exception, const std::string& reason)
			{
			PyErr_SetString (exception, reason.c_str());
			}
		
		Exception (PyObject* exception, Object &reason);		

		void clear() // clear the error
		// technically but not philosophically const
			{
			PyErr_Clear();
			}
		};
	
	
	// Abstract
	class StandardError: public Exception
		{
	protected: 
		explicit StandardError()
			{}
		};
	
	class LookupError: public StandardError
		{
	protected: 
		explicit LookupError()
			{}
		};
	
	class ArithmeticError: public StandardError
		{
	protected: 
		explicit ArithmeticError()
			{}
		};
	
	class EnvironmentError: public StandardError
		{
	protected: 
		explicit EnvironmentError()
			{}
		};
	
	// Concrete
	
	class TypeError: public StandardError
		{
	public:
		TypeError (const std::string& reason)
			: StandardError()
			{
			PyErr_SetString (Py::_Exc_TypeError(),reason.c_str());
			}
		};
	
	class IndexError: public LookupError
		{
	public:
		IndexError (const std::string& reason)
			: LookupError()
			{
			PyErr_SetString (Py::_Exc_IndexError(), reason.c_str());
			}
		};
	
	class AttributeError: public StandardError
		{
	public:
		AttributeError (const std::string& reason)
			: StandardError()
			{
			PyErr_SetString (Py::_Exc_AttributeError(), reason.c_str());
			}		
		};
	
	class NameError: public StandardError
		{
	public:
		NameError (const std::string& reason)
			: StandardError()
			{
			PyErr_SetString (Py::_Exc_NameError(), reason.c_str());
			}
		};
	
	class RuntimeError: public StandardError
		{
	public:
		RuntimeError (const std::string& reason)
			: StandardError()
			{
			PyErr_SetString (Py::_Exc_RuntimeError(), reason.c_str());
			}
		};
	
	class SystemError: public StandardError
		{
	public:
		SystemError (const std::string& reason)
			: StandardError()
			{
			PyErr_SetString (Py::_Exc_SystemError(),reason.c_str());
			}
		};
	
	class KeyError: public LookupError
		{
	public:
		KeyError (const std::string& reason)
			: LookupError()
			{
			PyErr_SetString (Py::_Exc_KeyError(),reason.c_str());
			}
		};
	
	
	class ValueError: public StandardError
		{
	public:
		ValueError (const std::string& reason)
			: StandardError()
			{
			PyErr_SetString (Py::_Exc_ValueError(), reason.c_str());
			}
		};
	
	class OverflowError: public ArithmeticError
		{
	public:
		OverflowError (const std::string& reason)
			: ArithmeticError()
			{
			PyErr_SetString (Py::_Exc_OverflowError(), reason.c_str());
			}		
		};
	
	class ZeroDivisionError: public ArithmeticError
		{
	public:
		ZeroDivisionError (const std::string& reason)
			: ArithmeticError() 
			{
			PyErr_SetString (Py::_Exc_ZeroDivisionError(), reason.c_str());
			}
		};
	
	class FloatingPointError: public ArithmeticError
		{
	public:
		FloatingPointError (const std::string& reason)
			: ArithmeticError() 
			{
			PyErr_SetString (Py::_Exc_FloatingPointError(), reason.c_str());
			}
		};
	
	class MemoryError: public StandardError
		{
	public:
		MemoryError (const std::string& reason)
			: StandardError()
			{
			PyErr_SetString (Py::_Exc_MemoryError(), reason.c_str());
			}	
		};
	
	class SystemExit: public StandardError
		{
	public:
		SystemExit (const std::string& reason)
			: StandardError() 
			{
			PyErr_SetString (Py::_Exc_SystemExit(),reason.c_str());
			}
		};

	}// Py

#endif
