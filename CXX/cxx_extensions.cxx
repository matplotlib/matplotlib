//-----------------------------------------------------------------------------
//
// Copyright (c) 1998 - 2007, The Regents of the University of California
// Produced at the Lawrence Livermore National Laboratory
// All rights reserved.
//
// This file is part of PyCXX. For details,see http://cxx.sourceforge.net/. The
// full copyright notice is contained in the file COPYRIGHT located at the root
// of the PyCXX distribution.
//
// Redistribution  and  use  in  source  and  binary  forms,  with  or  without
// modification, are permitted provided that the following conditions are met:
//
//  - Redistributions of  source code must  retain the above  copyright notice,
//    this list of conditions and the disclaimer below.
//  - Redistributions in binary form must reproduce the above copyright notice,
//    this  list of  conditions  and  the  disclaimer (as noted below)  in  the
//    documentation and/or materials provided with the distribution.
//  - Neither the name of the UC/LLNL nor  the names of its contributors may be
//    used to  endorse or  promote products derived from  this software without
//    specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT  HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR  IMPLIED WARRANTIES, INCLUDING,  BUT NOT  LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND  FITNESS FOR A PARTICULAR  PURPOSE
// ARE  DISCLAIMED.  IN  NO  EVENT  SHALL  THE  REGENTS  OF  THE  UNIVERSITY OF
// CALIFORNIA, THE U.S.  DEPARTMENT  OF  ENERGY OR CONTRIBUTORS BE  LIABLE  FOR
// ANY  DIRECT,  INDIRECT,  INCIDENTAL,  SPECIAL,  EXEMPLARY,  OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT  LIMITED TO, PROCUREMENT OF  SUBSTITUTE GOODS OR
// SERVICES; LOSS OF  USE, DATA, OR PROFITS; OR  BUSINESS INTERRUPTION) HOWEVER
// CAUSED  AND  ON  ANY  THEORY  OF  LIABILITY,  WHETHER  IN  CONTRACT,  STRICT
// LIABILITY, OR TORT  (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY  WAY
// OUT OF THE  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
// DAMAGE.
//
//-----------------------------------------------------------------------------
#include "CXX/Extensions.hxx"
#include "CXX/Exception.hxx"

#include <assert.h>

namespace Py 
{
//================================================================================
//
//    Implementation of MethodTable
//
//================================================================================

PyMethodDef MethodTable::method( const char* method_name, PyCFunction f, int flags, const char* doc ) 
{
    PyMethodDef m;
    m.ml_name = const_cast<char*>( method_name );
    m.ml_meth = f;
    m.ml_flags = flags;
    m.ml_doc = const_cast<char*>( doc );
    return m;
}

MethodTable::MethodTable()
{
    t.push_back( method( 0, 0, 0, 0 ) );
    mt = 0;
}

MethodTable::~MethodTable()
{
    delete [] mt;
}

void MethodTable::add( const char* method_name, PyCFunction f, const char* doc, int flag )
{
    if( !mt )
    {
        t.insert( t.end()-1, method( method_name, f, flag, doc ) );
    }
    else
    {
        throw RuntimeError( "Too late to add a module method!" );
    }
}

PyMethodDef* MethodTable::table()
{    
    if( !mt )
    {
        Py_ssize_t t1size = t.size();
        mt = new PyMethodDef[t1size];
        int j = 0;
        for( std::vector<PyMethodDef>::iterator i = t.begin(); i != t.end(); i++ )
        {
            mt[j++] = *i;
        }
    }
    return mt;
}

//================================================================================
//
//    Implementation of ExtensionModule
//
//================================================================================
ExtensionModuleBase::ExtensionModuleBase( const char *name )
    : module_name( name )
    , full_module_name( __Py_PackageContext() != NULL ? std::string( __Py_PackageContext() ) : module_name )
    , method_table()
{}

ExtensionModuleBase::~ExtensionModuleBase()
{}

const std::string &ExtensionModuleBase::name() const
{
    return module_name;
}

const std::string &ExtensionModuleBase::fullName() const
{
    return full_module_name;
}

class ExtensionModuleBasePtr : public PythonExtension<ExtensionModuleBasePtr>
{
public:
    ExtensionModuleBasePtr( ExtensionModuleBase *_module )
        : module( _module )
    {}
    virtual ~ExtensionModuleBasePtr()
    {}

    ExtensionModuleBase *module;
};


void ExtensionModuleBase::initialize( const char *module_doc )
{
    PyObject *module_ptr = new ExtensionModuleBasePtr( this );

    Py_InitModule4
    (
    const_cast<char *>( module_name.c_str() ),    // name
    method_table.table(),                // methods
    const_cast<char *>( module_doc ),        // docs
    module_ptr,                    // pass to functions as "self"
    PYTHON_API_VERSION                // API version
    );
}

Py::Module ExtensionModuleBase::module(void) const
{
    return Module( full_module_name );
}

Py::Dict ExtensionModuleBase::moduleDictionary(void) const
{
    return module().getDict();
}

//--------------------------------------------------------------------------------

//================================================================================
//
//    Implementation of PythonType
//
//================================================================================

extern "C"
{
    static void standard_dealloc(PyObject* p);
    //
    // All the following functions redirect the call from Python
    // onto the matching virtual function in PythonExtensionBase
    //
    static int print_handler (PyObject*, FILE *, int);
    static PyObject* getattr_handler (PyObject*, char*);
    static int setattr_handler (PyObject*, char*, PyObject*);
    static PyObject* getattro_handler (PyObject*, PyObject*);
    static int setattro_handler (PyObject*, PyObject*, PyObject*);
    static int compare_handler (PyObject*, PyObject*);
    static PyObject* repr_handler (PyObject*);
    static PyObject* str_handler (PyObject*);
    static long hash_handler (PyObject*);
    static PyObject* call_handler (PyObject*, PyObject*, PyObject*);
    static PyObject* iter_handler (PyObject*);
    static PyObject* iternext_handler (PyObject*);

    // Sequence methods
    static Py_ssize_t sequence_length_handler(PyObject*);
    static PyObject* sequence_concat_handler(PyObject*,PyObject*);
    static PyObject* sequence_repeat_handler(PyObject*, Py_ssize_t);
    static PyObject* sequence_item_handler(PyObject*, Py_ssize_t);
    static PyObject* sequence_slice_handler(PyObject*, Py_ssize_t, Py_ssize_t);
    static int sequence_ass_item_handler(PyObject*, Py_ssize_t, PyObject*);
    static int sequence_ass_slice_handler(PyObject*, Py_ssize_t, Py_ssize_t, PyObject*);
    // Mapping
    static Py_ssize_t mapping_length_handler(PyObject*);
    static PyObject* mapping_subscript_handler(PyObject*, PyObject*);
    static int mapping_ass_subscript_handler(PyObject*, PyObject*, PyObject*);

    // Numeric methods
    static int number_nonzero_handler (PyObject*);
    static PyObject* number_negative_handler (PyObject*);
    static PyObject* number_positive_handler (PyObject*);
    static PyObject* number_absolute_handler (PyObject*);
    static PyObject* number_invert_handler (PyObject*);
    static PyObject* number_int_handler (PyObject*);
    static PyObject* number_float_handler (PyObject*);
    static PyObject* number_long_handler (PyObject*);
    static PyObject* number_oct_handler (PyObject*);
    static PyObject* number_hex_handler (PyObject*);
    static PyObject* number_add_handler (PyObject*, PyObject*);
    static PyObject* number_subtract_handler (PyObject*, PyObject*);
    static PyObject* number_multiply_handler (PyObject*, PyObject*);
    static PyObject* number_divide_handler (PyObject*, PyObject*);
    static PyObject* number_remainder_handler (PyObject*, PyObject*);
    static PyObject* number_divmod_handler (PyObject*, PyObject*);
    static PyObject* number_lshift_handler (PyObject*, PyObject*);
    static PyObject* number_rshift_handler (PyObject*, PyObject*);
    static PyObject* number_and_handler (PyObject*, PyObject*);
    static PyObject* number_xor_handler (PyObject*, PyObject*);
    static PyObject* number_or_handler (PyObject*, PyObject*);
    static PyObject* number_power_handler(PyObject*, PyObject*, PyObject*);

    // Buffer
    static Py_ssize_t buffer_getreadbuffer_handler (PyObject*, Py_ssize_t, void**);
    static Py_ssize_t buffer_getwritebuffer_handler (PyObject*, Py_ssize_t, void**);
    static Py_ssize_t buffer_getsegcount_handler (PyObject*, Py_ssize_t*);
}


extern "C" void standard_dealloc( PyObject* p )
{
    PyMem_DEL( p );
}

PythonType & PythonType::supportSequenceType()
{
    if( !sequence_table )
    {
        sequence_table = new PySequenceMethods;
        memset( sequence_table, 0, sizeof( PySequenceMethods ) );   // ensure new fields are 0
        table->tp_as_sequence = sequence_table;
        sequence_table->sq_length = sequence_length_handler;
        sequence_table->sq_concat = sequence_concat_handler;
        sequence_table->sq_repeat = sequence_repeat_handler;
        sequence_table->sq_item = sequence_item_handler;
        sequence_table->sq_slice = sequence_slice_handler;

        sequence_table->sq_ass_item = sequence_ass_item_handler;    // BAS setup seperately?
        sequence_table->sq_ass_slice = sequence_ass_slice_handler;  // BAS setup seperately?
    }
    return *this;
}

PythonType & PythonType::supportMappingType()
{
    if( !mapping_table )
    {
        mapping_table = new PyMappingMethods;
        memset( mapping_table, 0, sizeof( PyMappingMethods ) );   // ensure new fields are 0
        table->tp_as_mapping = mapping_table;
        mapping_table->mp_length = mapping_length_handler;
        mapping_table->mp_subscript = mapping_subscript_handler;
        mapping_table->mp_ass_subscript = mapping_ass_subscript_handler;    // BAS setup seperately?
    }
    return *this;
}

PythonType & PythonType::supportNumberType()
{
    if( !number_table )
    {
        number_table = new PyNumberMethods;
        memset( number_table, 0, sizeof( PyNumberMethods ) );   // ensure new fields are 0
        table->tp_as_number = number_table;
        number_table->nb_add = number_add_handler;
        number_table->nb_subtract = number_subtract_handler;
        number_table->nb_multiply = number_multiply_handler;
        number_table->nb_divide = number_divide_handler;
        number_table->nb_remainder = number_remainder_handler;
        number_table->nb_divmod = number_divmod_handler;
        number_table->nb_power = number_power_handler;
        number_table->nb_negative = number_negative_handler;
        number_table->nb_positive = number_positive_handler;
        number_table->nb_absolute = number_absolute_handler;
        number_table->nb_nonzero = number_nonzero_handler;
        number_table->nb_invert = number_invert_handler;
        number_table->nb_lshift = number_lshift_handler;
        number_table->nb_rshift = number_rshift_handler;
        number_table->nb_and = number_and_handler;
        number_table->nb_xor = number_xor_handler;
        number_table->nb_or = number_or_handler;
        number_table->nb_coerce = 0;
        number_table->nb_int = number_int_handler;
        number_table->nb_long = number_long_handler;
        number_table->nb_float = number_float_handler;
        number_table->nb_oct = number_oct_handler;
        number_table->nb_hex = number_hex_handler;
    }
    return *this;
}

PythonType & PythonType::supportBufferType()
{
    if( !buffer_table )
    {
        buffer_table = new PyBufferProcs;
        memset( buffer_table, 0, sizeof( PyBufferProcs ) );   // ensure new fields are 0
        table->tp_as_buffer = buffer_table;
        buffer_table->bf_getreadbuffer = buffer_getreadbuffer_handler;
        buffer_table->bf_getwritebuffer = buffer_getwritebuffer_handler;
        buffer_table->bf_getsegcount = buffer_getsegcount_handler;
    }
    return *this;
}

// if you define one sequence method you must define 
// all of them except the assigns

PythonType::PythonType( size_t basic_size, int itemsize, const char *default_name )
    : table( new PyTypeObject )
    , sequence_table( NULL )
    , mapping_table( NULL )
    , number_table( NULL )
    , buffer_table( NULL )
{
    memset( table, 0, sizeof( PyTypeObject ) );   // ensure new fields are 0
    *reinterpret_cast<PyObject*>( table ) = py_object_initializer;
    table->ob_type = _Type_Type();
    table->ob_size = 0;
    table->tp_name = const_cast<char *>( default_name );
    table->tp_basicsize = basic_size;
    table->tp_itemsize = itemsize;
    table->tp_dealloc = ( destructor ) standard_dealloc;
    table->tp_print = 0;
    table->tp_getattr = 0;
    table->tp_setattr = 0;
    table->tp_compare = 0;
    table->tp_repr = 0;
    table->tp_as_number = 0;
    table->tp_as_sequence = 0;
    table->tp_as_mapping =  0;
    table->tp_hash = 0;
    table->tp_call = 0;
    table->tp_str = 0;
    table->tp_getattro = 0;
    table->tp_setattro = 0;
    table->tp_as_buffer = 0;
    table->tp_flags = Py_TPFLAGS_DEFAULT;
    table->tp_doc = 0;
#if PY_MAJOR_VERSION > 2 || (PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >= 0)
    // first use in 2.0
    table->tp_traverse = 0L;
    table->tp_clear = 0L;
#else
    table->tp_xxx5 = 0L;
    table->tp_xxx6 = 0L;
#endif
#if PY_MAJOR_VERSION > 2 || (PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >= 1)
    // first defined in 2.1
    table->tp_richcompare = 0L;
    table->tp_weaklistoffset = 0L;
#else
    table->tp_xxx7 = 0L;
    table->tp_xxx8 = 0L;
#endif

#if PY_MAJOR_VERSION > 2 || (PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION >= 2)
    // first defined in 2.3
    table->tp_iter = 0L;
    table->tp_iternext = 0L;
#endif

#ifdef COUNT_ALLOCS
    table->tp_alloc = 0;
    table->tp_free = 0;
    table->tp_maxalloc = 0;
    table->tp_next = 0;
#endif
}

PythonType::~PythonType( )
{
    delete table;
    delete sequence_table;
    delete mapping_table;
    delete number_table;
    delete buffer_table;
}

PyTypeObject* PythonType::type_object( ) const
{return table;}

PythonType & PythonType::name( const char* nam )
{
    table->tp_name = const_cast<char *>( nam );
    return *this;
}

const char *PythonType::getName() const
{
    return table->tp_name;
}

PythonType & PythonType::doc( const char* d )
{
    table->tp_doc = const_cast<char *>( d );
    return *this;
}

const char *PythonType::getDoc() const
{
    return table->tp_doc;
}

PythonType & PythonType::dealloc( void( *f )( PyObject* ))
{
    table->tp_dealloc = f;
    return *this;
}

PythonType & PythonType::supportPrint()
{
    table->tp_print = print_handler;
    return *this;
}

PythonType & PythonType::supportGetattr()
{
    table->tp_getattr = getattr_handler;
    return *this;
}

PythonType & PythonType::supportSetattr()
{
    table->tp_setattr = setattr_handler;
    return *this;
}

PythonType & PythonType::supportGetattro()
{
    table->tp_getattro = getattro_handler;
    return *this;
}

PythonType & PythonType::supportSetattro()
{
    table->tp_setattro = setattro_handler;
    return *this;
}

PythonType & PythonType::supportCompare()
{
    table->tp_compare = compare_handler;
    return *this;
}

PythonType & PythonType::supportRepr()
{
    table->tp_repr = repr_handler;
    return *this;
}

PythonType & PythonType::supportStr()
{
    table->tp_str = str_handler;
    return *this;
}

PythonType & PythonType::supportHash()
{
    table->tp_hash = hash_handler;
    return *this;
}

PythonType & PythonType::supportCall()
{
    table->tp_call = call_handler;
    return *this;
}

PythonType & PythonType::supportIter()
{
    table->tp_iter = iter_handler;
    table->tp_iternext = iternext_handler;
    return *this;
}

//--------------------------------------------------------------------------------
//
//    Handlers
//
//--------------------------------------------------------------------------------
extern "C" int print_handler( PyObject *self, FILE *fp, int flags )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return p->print( fp, flags );
    }
    catch( Py::Exception & )
    {
        return -1;    // indicate error
    }
}

extern "C" PyObject* getattr_handler( PyObject *self, char *name )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->getattr( name ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" int setattr_handler( PyObject *self, char *name, PyObject *value )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return p->setattr( name, Py::Object( value ) );
    }
    catch( Py::Exception & )
    {
        return -1;    // indicate error
    }
}

extern "C" PyObject* getattro_handler( PyObject *self, PyObject *name )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->getattro( Py::Object( name ) ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" int setattro_handler( PyObject *self, PyObject *name, PyObject *value )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return p->setattro( Py::Object( name ), Py::Object( value ) );
    }
    catch( Py::Exception & )
    {
        return -1;    // indicate error
    }
}

extern "C" int compare_handler( PyObject *self, PyObject *other )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return p->compare( Py::Object( other ) );
    }
    catch( Py::Exception & )
    {
        return -1;    // indicate error
    }
}

extern "C" PyObject* repr_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->repr() );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* str_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->str() );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" long hash_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return p->hash();
    }
    catch( Py::Exception & )
    {
        return -1;    // indicate error
    }
}

extern "C" PyObject* call_handler( PyObject *self, PyObject *args, PyObject *kw )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        if( kw != NULL )
            return new_reference_to( p->call( Py::Object( args ), Py::Object( kw ) ) );
        else
            return new_reference_to( p->call( Py::Object( args ), Py::Object() ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* iter_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->iter() );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* iternext_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return p->iternext();  // might be a NULL ptr on end of iteration
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}


// Sequence methods
extern "C" Py_ssize_t sequence_length_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return p->sequence_length();
    }
    catch( Py::Exception & )
    {
        return -1;    // indicate error
    }
}

extern "C" PyObject* sequence_concat_handler( PyObject *self, PyObject *other )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->sequence_concat( Py::Object( other ) ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* sequence_repeat_handler( PyObject *self, Py_ssize_t count )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->sequence_repeat( count ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* sequence_item_handler( PyObject *self, Py_ssize_t index )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->sequence_item( index ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* sequence_slice_handler( PyObject *self, Py_ssize_t first, Py_ssize_t last )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->sequence_slice( first, last ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" int sequence_ass_item_handler( PyObject *self, Py_ssize_t index, PyObject *value )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return p->sequence_ass_item( index, Py::Object( value ) );
    }
    catch( Py::Exception & )
    {
        return -1;    // indicate error
    }
}

extern "C" int sequence_ass_slice_handler( PyObject *self, Py_ssize_t first, Py_ssize_t last, PyObject *value )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return p->sequence_ass_slice( first, last, Py::Object( value ) );
    }
    catch( Py::Exception & )
    {
        return -1;    // indicate error
    }
}

// Mapping
extern "C" Py_ssize_t mapping_length_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return p->mapping_length();
    }
    catch( Py::Exception & )
    {
        return -1;    // indicate error
    }
}

extern "C" PyObject* mapping_subscript_handler( PyObject *self, PyObject *key )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->mapping_subscript( Py::Object( key ) ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" int mapping_ass_subscript_handler( PyObject *self, PyObject *key, PyObject *value )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return p->mapping_ass_subscript( Py::Object( key ), Py::Object( value ) );
    }
    catch( Py::Exception & )
    {
        return -1;    // indicate error
    }
}

// Number
extern "C" int number_nonzero_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return p->number_nonzero();
    }
    catch( Py::Exception & )
    {
        return -1;    // indicate error
    }
}

extern "C" PyObject* number_negative_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_negative() );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_positive_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_positive() );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_absolute_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_absolute() );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_invert_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_invert() );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_int_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_int() );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_float_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_float() );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_long_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_long() );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_oct_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_oct() );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_hex_handler( PyObject *self )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_hex() );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_add_handler( PyObject *self, PyObject *other )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_add( Py::Object( other ) ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_subtract_handler( PyObject *self, PyObject *other )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_subtract( Py::Object( other ) ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_multiply_handler( PyObject *self, PyObject *other )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_multiply( Py::Object( other ) ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_divide_handler( PyObject *self, PyObject *other )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_divide( Py::Object( other ) ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_remainder_handler( PyObject *self, PyObject *other )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_remainder( Py::Object( other ) ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_divmod_handler( PyObject *self, PyObject *other )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_divmod( Py::Object( other ) ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_lshift_handler( PyObject *self, PyObject *other )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_lshift( Py::Object( other ) ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_rshift_handler( PyObject *self, PyObject *other )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_rshift( Py::Object( other ) ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_and_handler( PyObject *self, PyObject *other )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_and( Py::Object( other ) ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_xor_handler( PyObject *self, PyObject *other )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_xor( Py::Object( other ) ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_or_handler( PyObject *self, PyObject *other )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_or( Py::Object( other ) ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

extern "C" PyObject* number_power_handler( PyObject *self, PyObject *x1, PyObject *x2 )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return new_reference_to( p->number_power( Py::Object( x1 ), Py::Object( x2 ) ) );
    }
    catch( Py::Exception & )
    {
        return NULL;    // indicate error
    }
}

// Buffer
extern "C" Py_ssize_t buffer_getreadbuffer_handler( PyObject *self, Py_ssize_t index, void **pp )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return p->buffer_getreadbuffer( index, pp );
    }
    catch( Py::Exception & )
    {
        return -1;    // indicate error
    }
}

extern "C" Py_ssize_t buffer_getwritebuffer_handler( PyObject *self, Py_ssize_t index, void **pp )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return p->buffer_getwritebuffer( index, pp );
    }
    catch( Py::Exception & )
    {
        return -1;    // indicate error
    }
}

extern "C" Py_ssize_t buffer_getsegcount_handler( PyObject *self, Py_ssize_t *count )
{
    try
    {
        PythonExtensionBase *p = static_cast<PythonExtensionBase *>( self );
        return p->buffer_getsegcount( count );
    }
    catch( Py::Exception & )
    {
        return -1;    // indicate error
    }
}


//================================================================================
//
//    Implementation of PythonExtensionBase
//
//================================================================================
#define missing_method( method ) \
throw RuntimeError( "Extension object does not support method " #method );

PythonExtensionBase::PythonExtensionBase()
{
}

PythonExtensionBase::~PythonExtensionBase()
{
    assert( ob_refcnt == 0 );
}

int PythonExtensionBase::print( FILE *, int )
{ missing_method( print ); return -1; }

int PythonExtensionBase::setattr( const char*, const Py::Object & )
{ missing_method( setattr ); return -1; }

Py::Object PythonExtensionBase::getattro( const Py::Object & )
{ missing_method( getattro ); return Py::Nothing(); }

int PythonExtensionBase::setattro( const Py::Object &, const Py::Object & )
{ missing_method( setattro ); return -1; }

int PythonExtensionBase::compare( const Py::Object & )
{ missing_method( compare ); return -1; }

Py::Object PythonExtensionBase::repr()
{ missing_method( repr ); return Py::Nothing(); }

Py::Object PythonExtensionBase::str()
{ missing_method( str ); return Py::Nothing(); }

long PythonExtensionBase::hash()
{ missing_method( hash ); return -1; }

Py::Object PythonExtensionBase::call( const Py::Object &, const Py::Object & )
{ missing_method( call ); return Py::Nothing(); }

Py::Object PythonExtensionBase::iter()
{ missing_method( iter ); return Py::Nothing(); }

PyObject* PythonExtensionBase::iternext()
{ missing_method( iternext ); return NULL; }


// Sequence methods
int PythonExtensionBase::sequence_length()
{ missing_method( sequence_length ); return -1; }

Py::Object PythonExtensionBase::sequence_concat( const Py::Object & )
{ missing_method( sequence_concat ); return Py::Nothing(); }

Py::Object PythonExtensionBase::sequence_repeat( Py_ssize_t )
{ missing_method( sequence_repeat ); return Py::Nothing(); }

Py::Object PythonExtensionBase::sequence_item( Py_ssize_t )
{ missing_method( sequence_item ); return Py::Nothing(); }

Py::Object PythonExtensionBase::sequence_slice( Py_ssize_t, Py_ssize_t )
{ missing_method( sequence_slice ); return Py::Nothing(); }

int PythonExtensionBase::sequence_ass_item( Py_ssize_t, const Py::Object & )
{ missing_method( sequence_ass_item ); return -1; }

int PythonExtensionBase::sequence_ass_slice( Py_ssize_t, Py_ssize_t, const Py::Object & )
{ missing_method( sequence_ass_slice ); return -1; }


// Mapping
int PythonExtensionBase::mapping_length()
{ missing_method( mapping_length ); return -1; }

Py::Object PythonExtensionBase::mapping_subscript( const Py::Object & )
{ missing_method( mapping_subscript ); return Py::Nothing(); }

int PythonExtensionBase::mapping_ass_subscript( const Py::Object &, const Py::Object & )
{ missing_method( mapping_ass_subscript ); return -1; }


// Number
int PythonExtensionBase::number_nonzero()
{ missing_method( number_nonzero ); return -1; }

Py::Object PythonExtensionBase::number_negative()
{ missing_method( number_negative ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_positive()
{ missing_method( number_positive ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_absolute()
{ missing_method( number_absolute ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_invert()
{ missing_method( number_invert ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_int()
{ missing_method( number_int ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_float()
{ missing_method( number_float ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_long()
{ missing_method( number_long ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_oct()
{ missing_method( number_oct ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_hex()
{ missing_method( number_hex ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_add( const Py::Object & )
{ missing_method( number_add ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_subtract( const Py::Object & )
{ missing_method( number_subtract ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_multiply( const Py::Object & )
{ missing_method( number_multiply ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_divide( const Py::Object & )
{ missing_method( number_divide ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_remainder( const Py::Object & )
{ missing_method( number_remainder ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_divmod( const Py::Object & )
{ missing_method( number_divmod ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_lshift( const Py::Object & )
{ missing_method( number_lshift ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_rshift( const Py::Object & )
{ missing_method( number_rshift ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_and( const Py::Object & )
{ missing_method( number_and ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_xor( const Py::Object & )
{ missing_method( number_xor ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_or( const Py::Object & )
{ missing_method( number_or ); return Py::Nothing(); }

Py::Object PythonExtensionBase::number_power( const Py::Object &, const Py::Object & )
{ missing_method( number_power ); return Py::Nothing(); }


// Buffer
Py_ssize_t PythonExtensionBase::buffer_getreadbuffer( Py_ssize_t, void** )
{ missing_method( buffer_getreadbuffer ); return -1; }

Py_ssize_t PythonExtensionBase::buffer_getwritebuffer( Py_ssize_t, void** )
{ missing_method( buffer_getwritebuffer ); return -1; }

Py_ssize_t PythonExtensionBase::buffer_getsegcount( Py_ssize_t* )
{ missing_method( buffer_getsegcount ); return -1; }

//--------------------------------------------------------------------------------
//
//    Method call handlers for
//        PythonExtensionBase
//        ExtensionModuleBase
//
//--------------------------------------------------------------------------------

extern "C" PyObject *method_keyword_call_handler( PyObject *_self_and_name_tuple, PyObject *_args, PyObject *_keywords )
{
    try
    {
        Tuple self_and_name_tuple( _self_and_name_tuple );

        PyObject *self_in_cobject = self_and_name_tuple[0].ptr();
        void *self_as_void = PyCObject_AsVoidPtr( self_in_cobject );
        if( self_as_void == NULL )
            return NULL;

        ExtensionModuleBase *self = static_cast<ExtensionModuleBase *>( self_as_void );

        String py_name( self_and_name_tuple[1] );
        std::string name( py_name.as_std_string() );

        Tuple args( _args );
        if( _keywords == NULL )
        {
            Dict keywords;    // pass an empty dict

            Object result( self->invoke_method_keyword( name, args, keywords ) );
            return new_reference_to( result.ptr() );
        }

        Dict keywords( _keywords );

        Object result( self->invoke_method_keyword( name, args, keywords ) );
        return new_reference_to( result.ptr() );
    }
    catch( Exception & )
    {
        return 0;
    }
}

extern "C" PyObject *method_varargs_call_handler( PyObject *_self_and_name_tuple, PyObject *_args )
{
    try
    {
        Tuple self_and_name_tuple( _self_and_name_tuple );

        PyObject *self_in_cobject = self_and_name_tuple[0].ptr();
        void *self_as_void = PyCObject_AsVoidPtr( self_in_cobject );
        if( self_as_void == NULL )
        return NULL;

        ExtensionModuleBase *self = static_cast<ExtensionModuleBase *>( self_as_void );

        String py_name( self_and_name_tuple[1] );
        std::string name( py_name.as_std_string() );

        Tuple args( _args );

        Object result( self->invoke_method_varargs( name, args ) );

        return new_reference_to( result.ptr() );
    }
    catch( Exception & )
    {
        return 0;
    }
}

extern "C" void do_not_dealloc( void * )
{}


//--------------------------------------------------------------------------------
//
//    ExtensionExceptionType
//
//--------------------------------------------------------------------------------
ExtensionExceptionType::ExtensionExceptionType()
    : Py::Object()
{
}

void ExtensionExceptionType::init( ExtensionModuleBase &module, const std::string& name )
{
    std::string module_name( module.fullName() );
    module_name += ".";
    module_name += name;

    set( PyErr_NewException( const_cast<char *>( module_name.c_str() ), NULL, NULL ), true );
}

void ExtensionExceptionType::init( ExtensionModuleBase &module, const std::string& name, ExtensionExceptionType &parent)
 {
     std::string module_name( module.fullName() );
     module_name += ".";
     module_name += name;

    set( PyErr_NewException( const_cast<char *>( module_name.c_str() ), parent.ptr(), NULL ), true );
}
 
ExtensionExceptionType::~ExtensionExceptionType()
{
}

Exception::Exception( ExtensionExceptionType &exception, const std::string& reason )
{
    PyErr_SetString (exception.ptr(), reason.c_str());
}

Exception::Exception( ExtensionExceptionType &exception, Object &reason )
{
    PyErr_SetObject (exception.ptr(), reason.ptr());
}

Exception::Exception( PyObject* exception, Object &reason )
{
    PyErr_SetObject (exception, reason.ptr());
}        

}    // end of namespace Py
