//----------------------------------*-C++-*----------------------------------//
// Copyright 1998 The Regents of the University of California. 
// All rights reserved. See LEGAL.LLNL for full text and disclaimer.
//---------------------------------------------------------------------------//

#ifndef __CXX_Extensions__h
#define __CXX_Extensions__h


#ifdef _MSC_VER
// disable warning C4786: symbol greater than 255 character,
// okay to ignore
#pragma warning(disable: 4786)
#endif


#include "CXX/Config.hxx"
#include "CXX/Objects.hxx"

extern "C"
	{
	extern PyObject py_object_initializer;
	}

#include <vector>
#include <map>

namespace Py
	{
	class ExtensionModuleBase;
	
	// Make an Exception Type for use in raising custom exceptions
	class ExtensionExceptionType : public Object
		{
	public:
		ExtensionExceptionType();
		virtual ~ExtensionExceptionType();

		// call init to create the type
		void init(  ExtensionModuleBase &module, const std::string& name );
		};

	
	class MethodTable 
		{
	public:
		MethodTable();
		virtual ~MethodTable();
		
		void add(const char* method_name, PyCFunction f, const char* doc="", int flag=1);
		PyMethodDef* table();
		
	protected:
		std::vector<PyMethodDef> t;	// accumulator of PyMethodDef's
		PyMethodDef *mt;		// Actual method table produced when full
		
		static PyMethodDef method (const char* method_name, PyCFunction f, int flags = 1, const char* doc="");
		
	private:
		//
		// prevent the compiler generating these unwanted functions
		//
		MethodTable(const MethodTable& m);	//unimplemented
		void operator=(const MethodTable& m);	//unimplemented
		
		}; // end class MethodTable
	
	extern "C"
		{
		typedef PyObject *(*method_varargs_call_handler_t)( PyObject *_self, PyObject *_args );
		typedef PyObject *(*method_keyword_call_handler_t)( PyObject *_self, PyObject *_args, PyObject *_dict );
		};
	
	template<class T>
	class MethodDefExt : public PyMethodDef
		{
	public:
		typedef Object (T::*method_varargs_function_t)( const Tuple &args );
		typedef Object (T::*method_keyword_function_t)( const Tuple &args, const Dict &kws );
		
		MethodDefExt
		(
		const char *_name,
		method_varargs_function_t _function,
		method_varargs_call_handler_t _handler,
		const char *_doc
		)
			{
			ext_meth_def.ml_name = const_cast<char *>(_name);
			ext_meth_def.ml_meth = _handler;
			ext_meth_def.ml_flags = METH_VARARGS;
			ext_meth_def.ml_doc = const_cast<char *>(_doc);
			
			ext_varargs_function = _function;
			ext_keyword_function = NULL;
			}
		
		MethodDefExt
		(
		const char *_name,
		method_keyword_function_t _function,
		method_keyword_call_handler_t _handler,
		const char *_doc
		)
			{
			ext_meth_def.ml_name = const_cast<char *>(_name);
			ext_meth_def.ml_meth = method_varargs_call_handler_t( _handler );
			ext_meth_def.ml_flags = METH_VARARGS|METH_KEYWORDS;
			ext_meth_def.ml_doc = const_cast<char *>(_doc);
			
			ext_varargs_function = NULL;
			ext_keyword_function = _function;
			}
		
		~MethodDefExt()
			{}
		
		PyMethodDef ext_meth_def;
		method_varargs_function_t ext_varargs_function;	
		method_keyword_function_t ext_keyword_function;	
		};
	
	class ExtensionModuleBase
		{
	public:
		ExtensionModuleBase( const char *name );
		virtual ~ExtensionModuleBase();
		
		Module module(void) const;		// only valid after initialize() has been called
		Dict moduleDictionary(void) const;	// only valid after initialize() has been called
		
		virtual Object invoke_method_keyword( const std::string &_name, const Tuple &_args, const Dict &_keywords ) = 0;
		virtual Object invoke_method_varargs( const std::string &_name, const Tuple &_args ) = 0;
		
		const std::string &name() const;
		const std::string &fullName() const;
	
	protected:
		// Initialize the module
		void initialize( const char *module_doc );
		
		const std::string module_name;
		const std::string full_module_name;
		MethodTable method_table;
		
	private:
		
		//
		// prevent the compiler generating these unwanted functions
		//
		ExtensionModuleBase( const ExtensionModuleBase & );	//unimplemented
		void operator=( const ExtensionModuleBase & );		//unimplemented
		
		};
	
	extern "C" PyObject *method_keyword_call_handler( PyObject *_self_and_name_tuple, PyObject *_args, PyObject *_keywords );
	extern "C" PyObject *method_varargs_call_handler( PyObject *_self_and_name_tuple, PyObject *_args );
	extern "C" void do_not_dealloc( void * );
	
	
	template<TEMPLATE_TYPENAME T>
	class ExtensionModule : public ExtensionModuleBase
		{
	public:
		ExtensionModule( const char *name )
			: ExtensionModuleBase( name )
			{}
		virtual ~ExtensionModule()
			{}
		
	protected:
		typedef Object (T::*method_varargs_function_t)( const Tuple &args );
		typedef Object (T::*method_keyword_function_t)( const Tuple &args, const Dict &kws );
		typedef std::map<std::string,MethodDefExt<T> *> method_map_t;
		
		static void add_varargs_method( const char *name, method_varargs_function_t function, const char *doc="" )
			{
			method_map_t &mm = methods();
			
			MethodDefExt<T> *method_definition = new MethodDefExt<T>
			(
			name,
			function,
			method_varargs_call_handler,
			doc
			);
			
			mm[std::string( name )] = method_definition;
			}
		
		static void add_keyword_method( const char *name, method_keyword_function_t function, const char *doc="" )
			{
			method_map_t &mm = methods();
			
			MethodDefExt<T> *method_definition = new MethodDefExt<T>
			(
			name,
			function,
			method_keyword_call_handler,
			doc
			);
			
			mm[std::string( name )] = method_definition;
			}

		void initialize( const char *module_doc="" )
			{
			ExtensionModuleBase::initialize( module_doc );
			Dict dict( moduleDictionary() );
			
			//
			// put each of the methods into the modules dictionary
			// so that we get called back at the function in T.
			//
			method_map_t &mm = methods();
			EXPLICIT_TYPENAME method_map_t::iterator i;
			
			for( i=mm.begin(); i != mm.end(); ++i )
				{
				MethodDefExt<T> *method_definition = (*i).second;
				
				static PyObject *self = PyCObject_FromVoidPtr( this, do_not_dealloc );
				
				Tuple args( 2 );
				args[0] = Object( self );
				args[1] = String( (*i).first );
				
				PyObject *func = PyCFunction_New
				(
				&method_definition->ext_meth_def,
				new_reference_to( args )
				);
				
				dict[ (*i).first ] = Object( func );
				}
			}
		
	protected:	// Tom Malcolmson reports that derived classes need access to these
		
		static method_map_t &methods(void)
			{
			static method_map_t *map_of_methods = NULL;
			if( map_of_methods == NULL )
			map_of_methods = new method_map_t;
			
			return *map_of_methods;
			}
		
		
		// this invoke function must be called from within a try catch block
		virtual Object invoke_method_keyword( const std::string &name, const Tuple &args, const Dict &keywords )
			{
			method_map_t &mm = methods();
			MethodDefExt<T> *meth_def = mm[ name ];
			if( meth_def == NULL )
				{
				std::string error_msg( "CXX - cannot invoke keyword method named " );
				error_msg += name;
				throw RuntimeError( error_msg );
				}
			
			// cast up to the derived class
			T *self = static_cast<T *>(this);
			
			return (self->*meth_def->ext_keyword_function)( args, keywords );
			}
		
		// this invoke function must be called from within a try catch block
		virtual Object invoke_method_varargs( const std::string &name, const Tuple &args )
			{
			method_map_t &mm = methods();
			MethodDefExt<T> *meth_def = mm[ name ];
			if( meth_def == NULL )
				{
				std::string error_msg( "CXX - cannot invoke varargs method named " );
				error_msg += name;
				throw RuntimeError( error_msg );
				}
			
			// cast up to the derived class
			T *self = static_cast<T *>(this);
			
			return (self->*meth_def->ext_varargs_function)( args );
			}
		
	private:
		//
		// prevent the compiler generating these unwanted functions
		//
		ExtensionModule( const ExtensionModule<T> & );	//unimplemented
		void operator=( const ExtensionModule<T> & );	//unimplemented
		};
	
	
	class PythonType
		{
	public:
		// if you define one sequence method you must define 
		// all of them except the assigns
		
		PythonType (size_t base_size, int itemsize, const char *default_name );
		virtual ~PythonType ();
		
		const char *getName () const;
		const char *getDoc () const;

		PyTypeObject* type_object () const;
		void name (const char* nam);
		void doc (const char* d);
		void dealloc(void (*f)(PyObject*));
		
		void supportPrint(void);
		void supportGetattr(void);
		void supportSetattr(void);
		void supportGetattro(void);
		void supportSetattro(void);
		void supportCompare(void);
		void supportRepr(void);
		void supportStr(void);
		void supportHash(void);
		void supportCall(void);
		
		void supportSequenceType(void);
		void supportMappingType(void);
		void supportNumberType(void);
		void supportBufferType(void);
		
	protected:
		PyTypeObject		*table;
		PySequenceMethods	*sequence_table;
		PyMappingMethods	*mapping_table;
		PyNumberMethods		*number_table;
		PyBufferProcs		*buffer_table;
		
		void init_sequence();
		void init_mapping();
		void init_number();
		void init_buffer();
		
	private:
		//
		// prevent the compiler generating these unwanted functions
		//
		PythonType (const PythonType& tb);	// unimplemented
		void operator=(const PythonType& t);	// unimplemented
		
		}; // end of PythonType
	
	
	
	// Class PythonExtension is what you inherit from to create
	// a new Python extension type. You give your class itself
	// as the template paramter.
	
	// There are two ways that extension objects can get destroyed.
	// 1. Their reference count goes to zero
	// 2. Someone does an explicit delete on a pointer.
	// In (1) the problem is to get the destructor called 
	//        We register a special deallocator in the Python type object
	//        (see behaviors()) to do this.
	// In (2) there is no problem, the dtor gets called.
	
	// PythonExtension does not use the usual Python heap allocator, 
	// instead using new/delete. We do the setting of the type object
	// and reference count, usually done by PyObject_New, in the 
	// base class ctor.
	
	// This special deallocator does a delete on the pointer.
	
	
	class PythonExtensionBase : public PyObject
		{
	public:
		PythonExtensionBase();
		virtual ~PythonExtensionBase();
		
	public:
		virtual int print( FILE *, int );
		virtual Object getattr( const char * ) = 0;
		virtual int setattr( const char *, const Object & );
		virtual Object getattro( const Object & );
		virtual int setattro( const Object &, const Object & );
		virtual int compare( const Object & );
		virtual Object repr();
		virtual Object str();
		virtual long hash();
		virtual Object call( const Object &, const Object & );
		
		// Sequence methods
		virtual int sequence_length();
		virtual Object sequence_concat( const Object & );
		virtual Object sequence_repeat( int );
		virtual Object sequence_item( int );
		virtual Object sequence_slice( int, int );
		virtual int sequence_ass_item( int, const Object & );
		virtual int sequence_ass_slice( int, int, const Object & );
		
		// Mapping
		virtual int mapping_length();
		virtual Object mapping_subscript( const Object & );
		virtual int mapping_ass_subscript( const Object &, const Object & );
		
		// Number
		virtual int number_nonzero();
		virtual Object number_negative();
		virtual Object number_positive();
		virtual Object number_absolute();
		virtual Object number_invert();
		virtual Object number_int();
		virtual Object number_float();
		virtual Object number_long();
		virtual Object number_oct();
		virtual Object number_hex();
		virtual Object number_add( const Object & );
		virtual Object number_subtract( const Object & );
		virtual Object number_multiply( const Object & );
		virtual Object number_divide( const Object & );
		virtual Object number_remainder( const Object & );
		virtual Object number_divmod( const Object & );
		virtual Object number_lshift( const Object & );
		virtual Object number_rshift( const Object & );
		virtual Object number_and( const Object & );
		virtual Object number_xor( const Object & );
		virtual Object number_or( const Object & );
		virtual Object number_power( const Object &, const Object & );
		
		// Buffer
		virtual int buffer_getreadbuffer( int, void** );
		virtual int buffer_getwritebuffer( int, void** );
		virtual int buffer_getsegcount( int* );
		
	private:
		void missing_method( void );
		static PyObject *method_call_handler( PyObject *self, PyObject *args );
		};
	
	template<TEMPLATE_TYPENAME T>
	class PythonExtension: public PythonExtensionBase 
		{
	public:
		static PyTypeObject* type_object() 
			{
			return behaviors().type_object();
			}
		
		static int check( PyObject *p )
			{
			// is p like me?
			return p->ob_type == type_object();
			}
		
		static int check( const Object& ob )
			{
			return check( ob.ptr());
			}
		
		
		//
		// every object needs getattr implemented
		// to support methods
		//
		virtual Object getattr( const char *name )
			{
			return getattr_methods( name );
			}
		
	protected:
		explicit PythonExtension()
			: PythonExtensionBase()
			{
			#ifdef PyObject_INIT
			PyObject_INIT( this, type_object() );
			#else
			ob_refcnt = 1;
			ob_type = type_object();
			#endif
			
			// every object must support getattr
			behaviors().supportGetattr();
			}
		
		virtual ~PythonExtension()
			{} 
		
		static PythonType &behaviors()
			{
			static PythonType* p;
			if( p == NULL ) 
				{
#if defined( _CPPRTTI )
				const char *default_name = (typeid ( T )).name();
#else
				const char *default_name = "unknown";
#endif
				p = new PythonType( sizeof( T ), 0, default_name );
				p->dealloc( extension_object_deallocator );
				}
			
			return *p;
			}
		
		
		typedef Object (T::*method_varargs_function_t)( const Tuple &args );
		typedef Object (T::*method_keyword_function_t)( const Tuple &args, const Dict &kws );
		typedef std::map<std::string,MethodDefExt<T> *> method_map_t;
		
		// support the default attributes, __name__, __doc__ and methods
		virtual Object getattr_default( const char *_name )
			{
			std::string name( _name );

			if( name == "__name__" && type_object()->tp_name != NULL )
				{
				return Py::String( type_object()->tp_name );
				}
			else if( name == "__doc__" && type_object()->tp_doc != NULL )
				{
				return Py::String( type_object()->tp_doc );
				}

// trying to fake out being a class for help()
//			else if( name == "__bases__"  )
//				{
//				return Py::Tuple(0);
//				}
//			else if( name == "__module__"  )
//				{
//				return Py::Nothing();
//				}
//			else if( name == "__dict__"  )
//				{
//				return Py::Dict();
//				}
			else
				{
				return getattr_methods( _name );
				}
			}

		// turn a name into function object
		virtual Object getattr_methods( const char *_name )
			{
			std::string name( _name );
			
			method_map_t &mm = methods();
			
			if( name == "__methods__" )
				{
				List methods;
				
				for( EXPLICIT_TYPENAME method_map_t::iterator i = mm.begin(); i != mm.end(); ++i )
					methods.append( String( (*i).first ) );
				
				return methods;
				}
			
			// see if name exists
			if( mm.find( name ) == mm.end() )
				throw AttributeError( name );
			
			Tuple self( 2 );
			
			self[0] = Object( this );
			self[1] = String( name );
			
			MethodDefExt<T> *method_definition = mm[ name ];
			
			PyObject *func = PyCFunction_New( &method_definition->ext_meth_def, self.ptr() );
			
			return Object(func, true);
			}
		
		static void add_varargs_method( const char *name, method_varargs_function_t function, const char *doc="" )
			{
			method_map_t &mm = methods();
			
			MethodDefExt<T> *method_definition = new MethodDefExt<T>
			(
			name,
			function,
			method_varargs_call_handler,
			doc
			);
			
			mm[std::string( name )] = method_definition;
			}
		
		static void add_keyword_method( const char *name, method_keyword_function_t function, const char *doc="" )
			{
			method_map_t &mm = methods();
			
			MethodDefExt<T> *method_definition = new MethodDefExt<T>
			(
			name,
			function,
			method_keyword_call_handler,
			doc
			);
			
			mm[std::string( name )] = method_definition;
			}
		
	private:
		static method_map_t &methods(void)
			{
			static method_map_t *map_of_methods = NULL;
			if( map_of_methods == NULL )
			map_of_methods = new method_map_t;
			
			return *map_of_methods;
			}
		
		static PyObject *method_keyword_call_handler( PyObject *_self_and_name_tuple, PyObject *_args, PyObject *_keywords )
			{
			try
				{
				Tuple self_and_name_tuple( _self_and_name_tuple );
				
				PyObject *self_in_cobject = self_and_name_tuple[0].ptr();
				T *self = static_cast<T *>( self_in_cobject );
				
				String name( self_and_name_tuple[1] );
				
				method_map_t &mm = methods();
				MethodDefExt<T> *meth_def = mm[ name ];
				if( meth_def == NULL )
					return 0;
				
				Tuple args( _args );

				// _keywords may be NULL so be careful about the way the dict is created
				Dict keywords;
				if( _keywords != NULL )
					keywords = Dict( _keywords );
				
				Object result( (self->*meth_def->ext_keyword_function)( args, keywords ) );
				
				return new_reference_to( result.ptr() );
				}
			catch( Exception & )
				{
				return 0;
				}
			}
		
		static PyObject *method_varargs_call_handler( PyObject *_self_and_name_tuple, PyObject *_args )
			{
			try
				{
				Tuple self_and_name_tuple( _self_and_name_tuple );
				
				PyObject *self_in_cobject = self_and_name_tuple[0].ptr();
				T *self = static_cast<T *>( self_in_cobject );
				
				String name( self_and_name_tuple[1] );
				
				method_map_t &mm = methods();
				MethodDefExt<T> *meth_def = mm[ name ];
				if( meth_def == NULL )
					return 0;
				
				Tuple args( _args );
				
				Object result;
				
				// TMM: 7Jun'01 - Adding try & catch in case of STL debug-mode exceptions.
				#ifdef _STLP_DEBUG
				try
					{
					result = (self->*meth_def->ext_varargs_function)( args );
					}
				catch (std::__stl_debug_exception)
					{
					// throw cxx::RuntimeError( sErrMsg );
					throw cxx::RuntimeError( "Error message not set yet." );
					}
				#else
				result = (self->*meth_def->ext_varargs_function)( args );
				#endif // _STLP_DEBUG
				
				return new_reference_to( result.ptr() );
				}
			catch( Exception & )
				{
				return 0;
				}
			}
		
		static void extension_object_deallocator ( PyObject* t )
			{
			delete (T *)( t );
			}
		
		//
		// prevent the compiler generating these unwanted functions
		//
		explicit PythonExtension( const PythonExtension<T>& other );
		void operator=( const PythonExtension<T>& rhs );
		};
	
	//
	// ExtensionObject<T> is an Object that will accept only T's.
	//
	template<TEMPLATE_TYPENAME T>
	class ExtensionObject: public Object
		{
	public:
		
		explicit ExtensionObject ( PyObject *pyob )
			: Object( pyob )
			{
			validate();
			}
		
		ExtensionObject( const ExtensionObject<T>& other )
			: Object( *other )
			{
			validate();
			}
		
		ExtensionObject( const Object& other )
			: Object( *other )
			{
			validate();
			}
		
		ExtensionObject& operator= ( const Object& rhs )
			{
			return (*this = *rhs );
			}
		
		ExtensionObject& operator= ( PyObject* rhsp )
			{
			if( ptr() == rhsp )
			return *this;
			set( rhsp );
			return *this;
			}
		
		virtual bool accepts ( PyObject *pyob ) const
			{
			return ( pyob && T::check( pyob ));
			}       
		
		//
		//	Obtain a pointer to the PythonExtension object
		//
		T *extensionObject(void)
			{
			return static_cast<T *>( ptr() );
			}
		};
	
	} // Namespace Py
// End of CXX_Extensions.h
#endif
