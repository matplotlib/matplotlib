/******************************************************************************
* Copyright (c) 2005, Enthought, Inc.
* All rights reserved.
* 
* This software is provided without warranty under the terms of the BSD
* license included in enthought/LICENSE.txt and may be redistributed only
* under the conditions described in the aforementioned license.  The license
* is also available online at http://www.enthought.com/licenses/BSD.txt
* Thanks for using Enthought open source!
* 
* Author: David C. Morrill
* Date: 06/15/2004
* Description: C based implementation of the Traits package
******************************************************************************/

/*-----------------------------------------------------------------------------
|  Includes:
+----------------------------------------------------------------------------*/

#include "Python.h"
#include "structmember.h"

/*-----------------------------------------------------------------------------
|  Constants:
+----------------------------------------------------------------------------*/

static PyObject     * class_traits;    /* == "__class_traits__" */
static PyObject     * editor_property; /* == "editor" */
static PyObject     * class_prefix;    /* == "__prefix__" */
static PyObject     * empty_tuple;     /* == () */
static PyObject     * undefined;       /* Global 'undefined' value */
static PyObject     * TraitError;      /* TraitError exception */
static PyObject     * DelegationError; /* DelegationError exception */
static PyObject     * TraitListObject; /* TraitListObject class */
static PyObject     * TraitDictObject; /* TraitDictObject class */
static PyTypeObject * ctrait_type;     /* Python-level CTrait type reference */
static PyObject     * is_callable;     /* Marker for 'callable' value */
static PyObject     * _HasTraits_monitors; /* Object creation monitors. */

/*-----------------------------------------------------------------------------
|  Macro definitions:
+----------------------------------------------------------------------------*/

/* The following macro is automatically defined in Python 2.4 and later: */
#ifndef Py_VISIT
#define Py_VISIT(op) \
do { \
    if (op) { \
        int vret = visit((PyObject *)(op), arg);	\
        if (vret) return vret; \
    } \
} while (0)
#endif

/* The following macro is automatically defined in Python 2.4 and later: */
#ifndef Py_CLEAR
#define Py_CLEAR(op) \
do { \
    if (op) { \
        PyObject *tmp = (PyObject *)(op); \
        (op) = NULL;	 \
        Py_DECREF(tmp); \
    } \
} while (0)
#endif

#define DEFERRED_ADDRESS(ADDR) 0
#define PyTrait_CheckExact(op) ((op)->ob_type == ctrait_type)

#define PyHasTraits_Check(op) PyObject_TypeCheck(op, &has_traits_type)
#define PyHasTraits_CheckExact(op) ((op)->ob_type == &has_traits_type)

/* Trait method related: */

#define TP_DESCR_GET(t) \
    (PyType_HasFeature(t, Py_TPFLAGS_HAVE_CLASS) ? (t)->tp_descr_get : NULL)
#define OFF(x) offsetof(trait_method_object, x)

/* Field accessors: */
#define trait_method_GET_NAME(meth) \
        (((trait_method_object *) meth)->tm_name)
#define trait_method_GET_FUNCTION(meth) \
        (((trait_method_object *) meth)->tm_func)
#define trait_method_GET_SELF(meth) \
	(((trait_method_object *) meth)->tm_self)
#define trait_method_GET_TRAITS(meth) \
	(((trait_method_object *) meth)->tm_traits)
#define trait_method_GET_CLASS(meth) \
	(((trait_method_object *) meth)->tm_class)

/* Python version dependent macros: */
#if ( (PY_MAJOR_VERSION == 2) && (PY_MINOR_VERSION < 3) )
#define PyMODINIT_FUNC void
#define PyDoc_VAR(name) static char name[]
#define PyDoc_STRVAR(name,str) PyDoc_VAR(name) = PyDoc_STR(str)
#ifdef WITH_DOC_STRINGS
#define PyDoc_STR(str) str
#else
#define PyDoc_STR(str) ""
#endif
#endif
#if (PY_VERSION_HEX < 0x02050000)
typedef int Py_ssize_t;
#endif
    
/*-----------------------------------------------------------------------------
|  Forward declarations:
+----------------------------------------------------------------------------*/

static PyTypeObject trait_type;
static PyTypeObject trait_method_type;

/*-----------------------------------------------------------------------------
|  'ctraits' module doc string:
+----------------------------------------------------------------------------*/

PyDoc_STRVAR( ctraits__doc__,
"The ctraits module defines the CHasTraits and CTrait C extension types that\n"
"define the core performance oriented portions of the Traits package." );

/*-----------------------------------------------------------------------------
|  HasTraits behavior modification flags:
+----------------------------------------------------------------------------*/

/* Object has been initialized: */
#define HASTRAITS_INITED      0x00000001

/* Do not send notifications when a trait changes value: */
#define HASTRAITS_NO_NOTIFY   0x00000002

/* Requests that no event notifications be sent when this object is assigned to
   a trait: */
#define HASTRAITS_VETO_NOTIFY 0x00000004

/*-----------------------------------------------------------------------------
|  'CHasTraits' instance definition:
|
|  Note: traits are normally stored in the type's dictionary, but are added to 
|  the instance's traits dictionary 'trait_dict' when the traits are defined 
|  dynamically or 'on_trait_change' is called on an instance of the trait.
|  
|  All 'anytrait_changed' notification handlers are stored in the instance's
|  'notifiers' list.
+----------------------------------------------------------------------------*/

typedef struct {
    PyObject_HEAD               /* Standard Python object header */
	PyDictObject * ctrait_dict; /* Class traits dictionary */
	PyDictObject * itrait_dict; /* Instance traits dictionary */
    PyListObject * notifiers;   /* List of 'any trait changed' notification 
                                   handlers */
    int            flags;       /* Behavior modification flags */                                   
	PyObject     * obj_dict;    /* Object attribute dictionary ('__dict__') */
                                /* NOTE: 'obj_dict' field MUST be last field */
} has_traits_object;

static int call_notifiers ( PyListObject *, PyListObject *, 
                            has_traits_object *, PyObject *, PyObject *,
                            PyObject * new_value );

/*-----------------------------------------------------------------------------
|  'CTrait' flag values:
+----------------------------------------------------------------------------*/

/* The trait is a Property: */
#define TRAIT_PROPERTY        0x00000001

/* Should the delegate be modified (or the original object)? */ 
#define TRAIT_MODIFY_DELEGATE 0x00000002

/* Should a simple object identity test be performed (or a rich compare)? */
#define TRAIT_OBJECT_IDENTITY 0x00000004

/*-----------------------------------------------------------------------------
|  'CTrait' instance definition:
+----------------------------------------------------------------------------*/

typedef struct _trait_object a_trait_object;
typedef PyObject * (*trait_getattr)( a_trait_object *, has_traits_object *, 
                                     PyObject * );
typedef int (*trait_setattr)( a_trait_object *, a_trait_object *, 
                              has_traits_object *, PyObject *, PyObject * );
typedef int (*trait_post_setattr)( a_trait_object *, has_traits_object *, 
                                   PyObject *, PyObject * ); 
typedef PyObject * (*trait_validate)( a_trait_object *, has_traits_object *, 
                              PyObject *, PyObject * );
typedef PyObject * (*delegate_attr_name_func)( a_trait_object *, 
                                             has_traits_object *, PyObject * );                             

typedef struct _trait_object {
    PyObject_HEAD                    /* Standard Python object header */
    int                flags;        /* Flag bits */
    trait_getattr      getattr;      /* Get trait value handler */     
    trait_setattr      setattr;      /* Set trait value handler */
    trait_post_setattr post_setattr; /* Optional post 'setattr' handler */
    PyObject *         py_post_setattr; /* Python-based post 'setattr' hndlr */
    trait_validate     validate;     /* Validate trait value handler */
    PyObject *         py_validate;  /* Python-based validate value handler */
    int                default_value_type; /* Type of default value: see the
                                              'default_value_for' function */
    PyObject *         default_value;   /* Default value for trait */
    PyObject *         delegate_name;   /* Optional delegate name */
                                        /* Also used for 'property get' */
    PyObject *         delegate_prefix; /* Optional delegate prefix */
                                        /* Also used for 'property set' */
    delegate_attr_name_func delegate_attr_name; /* Optional routine to return*/
                                  /* the computed delegate attribute name */
    PyListObject *     notifiers; /* Optional list of notification handlers */
    PyObject *         handler;   /* Associated trait handler object */                                  
                                  /* NOTE: The 'obj_dict' field MUST be last */
    PyObject *         obj_dict;  /* Standard Python object dictionary */
} trait_object;

/* Forward declaration: */
static void trait_clone ( trait_object *, trait_object * );

/*-----------------------------------------------------------------------------
|  Raise a TraitError:
+----------------------------------------------------------------------------*/

static PyObject * 
raise_trait_error ( trait_object * trait, has_traits_object * obj, 
				    PyObject * name, PyObject * value ) {
                                              
    PyObject * result = PyObject_CallMethod( trait->handler,
                                          "error", "(OOO)", obj, name, value );
    Py_XDECREF( result );                                                  
    return NULL;
}    

/*-----------------------------------------------------------------------------
|  Raise a fatal trait error:
+----------------------------------------------------------------------------*/

static int
fatal_trait_error ( void ) {
    
    PyErr_SetString( TraitError, "Non-trait found in trait dictionary" );
    return -1;
}

/*-----------------------------------------------------------------------------
|  Raise an "attribute is not a string" error:
+----------------------------------------------------------------------------*/

static int
invalid_attribute_error ( void ) {
    
    PyErr_SetString( PyExc_TypeError, "attribute name must be string" );
    return -1;
}

/*-----------------------------------------------------------------------------
|  Raise an "invalid trait definition" error:
+----------------------------------------------------------------------------*/

static int
bad_trait_error ( void ) {
    
    PyErr_SetString( TraitError, "Invalid argument to trait constructor." );
    return -1;
}

/*-----------------------------------------------------------------------------
|  Raise an invalid delegate error:
+----------------------------------------------------------------------------*/

static int
bad_delegate_error ( has_traits_object * obj, PyObject * name ) {
    
    if ( PyString_Check( name ) ) {
        PyErr_Format( DelegationError,
            "The '%.400s' attribute of a '%.50s' object delegates to an attribute which is not a defined trait.",
	        PyString_AS_STRING( name ), obj->ob_type->tp_name );
        return -1;
    }
    return invalid_attribute_error();
}    

/*-----------------------------------------------------------------------------
|  Raise an invalid delegate error:
+----------------------------------------------------------------------------*/

static int
bad_delegate_error2 ( has_traits_object * obj, PyObject * name ) {
    
    if ( PyString_Check( name ) ) {
        PyErr_Format( DelegationError,
            "The '%.400s' attribute of a '%.50s' object has a delegate which does not have traits.",
	        PyString_AS_STRING( name ), obj->ob_type->tp_name );
        return -1;
    }
    return invalid_attribute_error();
}    

/*-----------------------------------------------------------------------------
|  Raise a delegation recursion error:
+----------------------------------------------------------------------------*/

static int
delegation_recursion_error ( has_traits_object * obj, PyObject * name ) {
    
    if ( PyString_Check( name ) ) {
        PyErr_Format( DelegationError,
	                  "Delegation recursion limit exceeded while setting the '%.400s' attribute of a '%.50s' object.",
	                  PyString_AS_STRING( name ), obj->ob_type->tp_name );
        return -1;
    }
    return invalid_attribute_error();
}    

static int
delegation_recursion_error2 ( has_traits_object * obj, PyObject * name ) {
    
    if ( PyString_Check( name ) ) {
        PyErr_Format( DelegationError,
	                  "Delegation recursion limit exceeded while getting the definition of the '%.400s' trait of a '%.50s' object.",
	                  PyString_AS_STRING( name ), obj->ob_type->tp_name );
        return -1;
    }
    return invalid_attribute_error();
}    

/*-----------------------------------------------------------------------------
|  Raise an attempt to delete read-only attribute error:
+----------------------------------------------------------------------------*/

static int
delete_readonly_error ( has_traits_object * obj, PyObject * name ) {
    
    if ( PyString_Check( name ) ) {
        PyErr_Format( TraitError,
	                  "Cannot delete the read only '%.400s' attribute of a '%.50s' object.",
	                  PyString_AS_STRING( name ), obj->ob_type->tp_name );
        return -1;
    }
    return invalid_attribute_error();
}    

/*-----------------------------------------------------------------------------
|  Raise an attempt to set a read-only attribute error:
+----------------------------------------------------------------------------*/

static int
set_readonly_error ( has_traits_object * obj, PyObject * name ) {
    
    if ( PyString_Check( name ) ) {
        PyErr_Format( TraitError,
	                  "Cannot modify the read only '%.400s' attribute of a '%.50s' object.",
	                  PyString_AS_STRING( name ), obj->ob_type->tp_name );
        return -1;
    }
    return invalid_attribute_error();
}    

/*-----------------------------------------------------------------------------
|  Raise an attempt to set an undefined attribute error:
+----------------------------------------------------------------------------*/

static int
set_disallow_error ( has_traits_object * obj, PyObject * name ) {
    
    if ( PyString_Check( name ) ) {
        PyErr_Format( TraitError,
	                  "Cannot set the undefined '%.400s' attribute of a '%.50s' object.",
	                  PyString_AS_STRING( name ), obj->ob_type->tp_name );
        return -1;
    }
    return invalid_attribute_error();
}  

/*-----------------------------------------------------------------------------
|  Raise an undefined attribute error:
+----------------------------------------------------------------------------*/

static void
unknown_attribute_error ( has_traits_object * obj, PyObject * name ) {
    
    PyErr_Format( PyExc_AttributeError,
                  "'%.50s' object has no attribute '%.400s'",
                  obj->ob_type->tp_name, PyString_AS_STRING( name ) );
}    

/*-----------------------------------------------------------------------------
|  Raise a '__dict__' must be set to a dictionary error:
+----------------------------------------------------------------------------*/

static int
dictionary_error ( void ) {
    
    PyErr_SetString( PyExc_TypeError, 
                     "__dict__ must be set to a dictionary." );
    return -1;
}

/*-----------------------------------------------------------------------------
|  Raise an exception when a trait method argument is of the wrong type:
+----------------------------------------------------------------------------*/

static PyObject * 
argument_error ( trait_object * trait, PyObject * meth, int arg, 
                 PyObject * obj, PyObject * name, PyObject * value ) {
    
    PyObject * arg_num = PyInt_FromLong( arg );
    if ( arg_num != NULL ) {
        PyObject * result = PyObject_CallMethod( trait->handler, 
                     "arg_error", "(OOOOO)", meth, arg_num, obj, name, value );
        Py_XDECREF( result );
        Py_XDECREF( arg_num );
    }
    return NULL;
}    

/*-----------------------------------------------------------------------------
|  Raise an exception when a trait method keyword argument is the wrong type:
+----------------------------------------------------------------------------*/

static PyObject * 
keyword_argument_error ( trait_object * trait, PyObject * meth, 
                         PyObject * obj, PyObject * name, PyObject * value ) {
    
    PyObject * result = PyObject_CallMethod( trait->handler, 
                           "keyword_error", "(OOOO)", meth, obj, name, value );
    Py_XDECREF( result );
    return NULL;
}    

/*-----------------------------------------------------------------------------
|  Raise an exception when a trait method keyword argument is the wrong type:
+----------------------------------------------------------------------------*/

static PyObject * 
dup_argument_error ( trait_object * trait, PyObject * meth, int arg, 
                     PyObject * obj, PyObject * name ) {
    
    PyObject * arg_num = PyInt_FromLong( arg );
    if ( arg_num != NULL ) {
        PyObject * result = PyObject_CallMethod( trait->handler, 
                       "dup_arg_error", "(OOOO)", meth, arg_num, obj, name );
        Py_XDECREF( result );
        Py_XDECREF( arg_num );
    }
    return NULL;
}    

/*-----------------------------------------------------------------------------
|  Raise an exception when a required trait method argument is missing:
+----------------------------------------------------------------------------*/

static PyObject * 
missing_argument_error ( trait_object * trait, PyObject * meth, int arg, 
                         PyObject * obj, PyObject * name ) {
    
    PyObject * arg_num = PyInt_FromLong( arg );
    if ( arg_num != NULL ) {
        PyObject * result = PyObject_CallMethod( trait->handler, 
                     "missing_arg_error", "(OOOO)", meth, arg_num, obj, name );
        Py_XDECREF( result );
        Py_XDECREF( arg_num );
    }
    return NULL;
}    

/*-----------------------------------------------------------------------------
|  Raise an exception when a required trait method argument is missing:
+----------------------------------------------------------------------------*/

static PyObject * 
too_may_args_error ( PyObject * name, int wanted, int received ) { 
    
    switch ( wanted ) {
        case 0:
            PyErr_Format( PyExc_TypeError,
                  "%.400s() takes no arguments (%.3d given)",
                  PyString_AS_STRING( name ), received );
            break;
        case 1:
            PyErr_Format( PyExc_TypeError,
                  "%.400s() takes exactly 1 argument (%.3d given)",
                  PyString_AS_STRING( name ), received );
            break;
        default:
            PyErr_Format( PyExc_TypeError,
                  "%.400s() takes exactly %.3d arguments (%.3d given)",
                  PyString_AS_STRING( name ), wanted, received );
            break;
    }
    return NULL;
}    

/*-----------------------------------------------------------------------------
|  Raise an exception when a trait method argument is of the wrong type:
+----------------------------------------------------------------------------*/

static void
invalid_result_error ( trait_object * trait, PyObject * meth, PyObject * obj,
                       PyObject * value ) {
    
    PyObject * result = PyObject_CallMethod( trait->handler, 
                                   "return_error", "(OOO)", meth, obj, value );
    Py_XDECREF( result );
}    

/*-----------------------------------------------------------------------------
|  Gets/Sets a possibly NULL (or callable) value:
+----------------------------------------------------------------------------*/

static PyObject * 
get_callable_value ( PyObject * value ) {
    PyObject * tuple, * temp;
    if ( value == NULL ) 
        value = Py_None;
    else if ( PyCallable_Check( value ) )
        value = is_callable;
    else if ( PyTuple_Check( value ) && 
              (PyInt_AsLong( PyTuple_GET_ITEM( value, 0 ) ) == 10) ) {
        tuple = PyTuple_New( 3 );
        if ( tuple != NULL ) {
            PyTuple_SET_ITEM( tuple, 0, temp = PyTuple_GET_ITEM( value, 0 ) );
            Py_INCREF( temp );
            PyTuple_SET_ITEM( tuple, 1, temp = PyTuple_GET_ITEM( value, 1 ) );
            Py_INCREF( temp );
            PyTuple_SET_ITEM( tuple, 2, is_callable );
            Py_INCREF( is_callable );
            value = tuple;
        }
    }              
    Py_INCREF( value );
    return value;
}

static PyObject * 
get_value ( PyObject * value ) {
    if ( value == NULL ) 
        value = Py_None;
    Py_INCREF( value );
    return value;
}

static int
set_value ( PyObject ** field, PyObject * value ) {
 
    Py_INCREF( value );
    Py_XDECREF( *field );
    *field = value;
    return 0;
}    

/*-----------------------------------------------------------------------------
|  Returns the result of calling a specified 'class' object with 1 argument:
+----------------------------------------------------------------------------*/

static PyObject *
call_class ( PyObject * class, trait_object * trait, has_traits_object * obj, 
             PyObject * name, PyObject * value ) {
 
    PyObject * result;
    
    PyObject * args = PyTuple_New( 4 );
    if ( args == NULL )
        return NULL;
    PyTuple_SET_ITEM( args, 0, trait->handler );
    PyTuple_SET_ITEM( args, 1, (PyObject *) obj );
    PyTuple_SET_ITEM( args, 2, name );
    PyTuple_SET_ITEM( args, 3, value );
    Py_INCREF( trait->handler );
    Py_INCREF( obj );
    Py_INCREF( name );
    Py_INCREF( value );
    result = PyObject_Call( class, args, NULL );
    Py_DECREF( args );
    return result;
}    

/*-----------------------------------------------------------------------------
|  Attempts to get the value of a key in a 'known to be a dictionary' object:
+----------------------------------------------------------------------------*/

static PyObject *
dict_getitem ( PyDictObject * dict, PyObject *key ) {
    
	long hash;
    
	assert( PyDict_Check( dict ) );  
    
	if ( !PyString_CheckExact( key ) ||
         ((hash = ((PyStringObject *) key)->ob_shash) == -1) ) {
		hash = PyObject_Hash( key );
		if ( hash == -1 ) {
			PyErr_Clear();
			return NULL;
		}
	}
	return (dict->ma_lookup)( dict, key, hash )->me_value;
}

/*-----------------------------------------------------------------------------
|  Gets the definition of the matching prefix based trait for a specified name:
|
|  - This should always return a trait definition unless a fatal Python error
|    occurs.
|  - The bulk of the work is delegated to a Python implemented method because
|    the implementation is complicated in C and does not need to be executed
|    very often relative to other operations.
|
+----------------------------------------------------------------------------*/

static trait_object *
get_prefix_trait ( has_traits_object * obj, PyObject * name ) {
    
    PyObject * trait = PyObject_CallMethod( (PyObject *) obj, 
                                            "__prefix_trait__", "(O)", name );
    if ( trait != NULL ) {
        assert( obj->ctrait_dict != NULL );
	    PyDict_SetItem( (PyObject *) obj->ctrait_dict, name, trait );
        Py_DECREF( trait );
    }
    return (trait_object *) trait;
}

/*-----------------------------------------------------------------------------
|  Handles the 'setattr' operation on a 'CHasTraits' instance:
+----------------------------------------------------------------------------*/

static int
has_traits_setattro ( has_traits_object * obj, 
                      PyObject          * name, 
                      PyObject          * value ) {
                          
    trait_object * trait;
    
    if ( (obj->itrait_dict == NULL) || 
         ((trait = (trait_object *) dict_getitem( obj->itrait_dict, name )) ==
           NULL) ) {
        trait = (trait_object *) dict_getitem( obj->ctrait_dict, name );
        if ( (trait == NULL) && 
             ((trait = get_prefix_trait( obj, name )) == NULL) )
            return -1;
    }
    return trait->setattr( trait, trait, obj, name, value );
}
/*-----------------------------------------------------------------------------
|  Allocates a CTrait instance:
+----------------------------------------------------------------------------*/

PyObject *
has_traits_new ( PyTypeObject * type, PyObject * args, PyObject * kwds ) {
    
    has_traits_object * obj = (has_traits_object *) type->tp_alloc( type, 0 );
    if ( obj != NULL ) {
        assert( type->tp_dict != NULL );
    	obj->ctrait_dict = (PyDictObject *) PyDict_GetItem( type->tp_dict, 
                                                            class_traits );
        assert( obj->ctrait_dict != NULL );
        assert( PyDict_Check( (PyObject *) obj->ctrait_dict ) );
        Py_INCREF( obj->ctrait_dict );
    }
    return (PyObject *) obj;
}    

int
has_traits_init ( PyObject * obj, PyObject * args, PyObject * kwds ) {
    
    PyObject * key;
    PyObject * value;
    PyObject * klass;
    PyObject * handler;
    PyObject * handler_args;
    int n;
    Py_ssize_t i = 0;
    
    /* Make sure no non-keyword arguments were specified: */
    if ( !PyArg_ParseTuple( args, "" ) )
        return -1;

    /* Set any traits specified in the constructor: */
    if ( kwds != NULL ) {
        while ( PyDict_Next( kwds, &i, &key, &value ) ) {
            if ( has_traits_setattro( (has_traits_object *)obj, key, value ) == -1 ) {
                return -1;
            }
        }
    }

    /* Notify any interested monitors that a new object has been created: */
    for ( i = 0, n = PyList_GET_SIZE( _HasTraits_monitors ); i < n; i++ ) {
        value = PyList_GET_ITEM( _HasTraits_monitors, i );
        assert( PyTuple_Check( value ) );
        assert( PyTuple_GET_SIZE( value ) == 2 );

        klass   = PyTuple_GET_ITEM( value, 0 );
        handler = PyTuple_GET_ITEM( value, 1 );

        if ( PyObject_IsInstance( obj, klass ) ) {
            handler_args = PyTuple_New( 1 );
            PyTuple_SetItem( handler_args, 0, obj );
            Py_INCREF( obj );
            PyObject_Call( handler, handler_args, NULL );
            Py_DECREF( handler_args );
        }
    }
    
    /* Indicate that the object has finished being initialized: */
    ((has_traits_object *)obj)->flags |= HASTRAITS_INITED;

    return 0;
}    

/*-----------------------------------------------------------------------------
|  Object clearing method:
+----------------------------------------------------------------------------*/

static int
has_traits_clear ( has_traits_object * obj ) {

    Py_CLEAR( obj->ctrait_dict );
    Py_CLEAR( obj->itrait_dict );
    Py_CLEAR( obj->notifiers );
    Py_CLEAR( obj->obj_dict );
    return 0;
}

/*-----------------------------------------------------------------------------
|  Deallocates an unused 'CHasTraits' instance: 
+----------------------------------------------------------------------------*/

static void 
has_traits_dealloc ( has_traits_object * obj ) {
    
    has_traits_clear( obj );
    obj->ob_type->tp_free( (PyObject *) obj );
}

/*-----------------------------------------------------------------------------
|  Garbage collector traversal method:
+----------------------------------------------------------------------------*/

static int
has_traits_traverse ( has_traits_object * obj, visitproc visit, void * arg ) {

    Py_VISIT( obj->ctrait_dict );
    Py_VISIT( obj->itrait_dict );
    Py_VISIT( obj->notifiers );
    Py_VISIT( obj->obj_dict );
	return 0;
}

/*-----------------------------------------------------------------------------
|  Handles the 'getattr' operation on a 'CHasTraits' instance:
+----------------------------------------------------------------------------*/

static PyObject *
has_traits_getattro ( has_traits_object * obj, PyObject * name ) {
    
    /* The following is a performance hack to short-circuit the normal
       look-up when the value is in the object's dictionary. */
	trait_object * trait;
	PyObject     * value;
    PyObject     * uname;
    long hash;
    
    PyDictObject * dict = (PyDictObject *) obj->obj_dict;
    
	if ( dict != NULL ) {
         assert( PyDict_Check( dict ) );
         if ( PyString_CheckExact( name ) ) {
              if ( (hash = ((PyStringObject *) name)->ob_shash) == -1 ) 
                  hash = PyObject_Hash( name );
	         value = (dict->ma_lookup)( dict, name, hash )->me_value;
             if ( value != NULL ) {
                 Py_INCREF( value );
                 return value;
             }
         } else {
            if ( PyString_Check( name ) ) {
	    	    hash = PyObject_Hash( name );
	    	    if ( hash == -1 ) 
	    		    return NULL;
        	    value = (dict->ma_lookup)( dict, name, hash )->me_value;
                if ( value != NULL ) {
                    Py_INCREF( value );
                    return value;
                }
            } else {
#ifdef Py_USING_UNICODE
                if ( PyUnicode_Check( name ) ) {
                    uname = PyUnicode_AsEncodedString( name, NULL, NULL );
                    if ( uname == NULL )
            		    return NULL;
                } else {
                    invalid_attribute_error();
                    return NULL;
                }
	    	    hash = PyObject_Hash( uname );
	    	    if ( hash == -1 ) {
                    Py_DECREF( uname );
	    		    return NULL;
                }
        	    value = (dict->ma_lookup)( dict, uname, hash )->me_value;
                Py_DECREF( uname );
                if ( value != NULL ) {
                    Py_INCREF( value );
                    return value;
                }
#else
                invalid_attribute_error();
                return NULL;
#endif
            }
         }
    }
    /* End of performance hack */

    if ( ((obj->itrait_dict != NULL) &&
         ((trait = (trait_object *) dict_getitem( obj->itrait_dict, name )) !=
          NULL)) ||
         ((trait = (trait_object *) dict_getitem( obj->ctrait_dict, name )) !=
          NULL) )
        return trait->getattr( trait, obj, name );
    
    if ( (value = PyObject_GenericGetAttr( (PyObject *) obj, name )) != NULL )
        return value;
    PyErr_Clear();
    if ( (trait = get_prefix_trait( obj, name )) != NULL )
        return trait->getattr( trait, obj, name );
    return NULL;
}


/*-----------------------------------------------------------------------------
|  Returns (and optionally creates) a specified instance or class trait:
+----------------------------------------------------------------------------*/

static PyObject *
get_trait ( has_traits_object * obj, PyObject * name, int instance ) {
 
    int i, n;
    PyDictObject * itrait_dict;
    trait_object * trait;
    trait_object * itrait;
    PyListObject * notifiers;
    PyListObject * inotifiers;
    PyObject     * item;
    
    /* If there already is an instance specific version of the requested trait,
       then return it: */
    itrait_dict = obj->itrait_dict;
    if ( itrait_dict != NULL ) {
        trait = (trait_object *) dict_getitem( itrait_dict, name );
        if ( trait != NULL ) {
            assert( PyTrait_CheckExact( trait ) );
            Py_INCREF( trait );
            return (PyObject *) trait;
        }
    }
    
    /* If only an instance trait can be returned (but not created), then 
       return None: */
    if ( instance == 1 ) {
        Py_INCREF( Py_None );
        return Py_None;
    }
    
    /* Otherwise, get the class specific version of the trait (creating a
       trait class version if necessary): */
    assert( obj->ctrait_dict != NULL );
    trait = (trait_object *) dict_getitem( obj->ctrait_dict, name );
    if ( trait == NULL ) {
        if ( instance == 0 ) {
            Py_INCREF( Py_None );
            return Py_None;
        }
        if ( (trait = get_prefix_trait( obj, name )) == NULL )
            return NULL;
    }
                                    
    assert( PyTrait_CheckExact( trait ) );
    
    /* If an instance specific trait is not needed, return the class trait: */
    if ( instance <= 0 ) {
        Py_INCREF( trait );
        return (PyObject *) trait;
    }
    
    /* Otherwise, create an instance trait dictionary if it does not exist: */
    if ( itrait_dict == NULL ) {
		obj->itrait_dict = itrait_dict = (PyDictObject *) PyDict_New();
		if ( itrait_dict == NULL )
            return NULL;
    }

    /* Create a new instance trait and clone the class trait into it: */
    itrait = (trait_object *) PyType_GenericAlloc( ctrait_type, 0 );
    trait_clone( itrait, trait );
    itrait->obj_dict = trait->obj_dict;
    Py_XINCREF( itrait->obj_dict );
    
    /* Copy the class trait's notifier list into the instance trait: */
    if ( (notifiers = trait->notifiers) != NULL ) {
        n = PyList_GET_SIZE( notifiers );
        itrait->notifiers = inotifiers = (PyListObject *) PyList_New( n );
        if ( inotifiers == NULL )
            return NULL;
        for ( i = 0; i < n; i++ ) {
            item = PyList_GET_ITEM( notifiers, i );
            PyList_SET_ITEM( inotifiers, i, item );
            Py_INCREF( item );
        }
    }
    
    /* Add the instance trait to the instance's trait dictionary and return
       the instance trait if successful: */
    if ( PyDict_SetItem( (PyObject *) itrait_dict, name, 
                         (PyObject *) itrait ) >= 0 )
        return (PyObject *) itrait;
    
    /* Otherwise, indicate that an error ocurred updating the dictionary: */
    return NULL;
}

/*-----------------------------------------------------------------------------
|  Returns (and optionally creates) a specified instance or class trait:
|
|  The legal values for 'instance' are:
|     2: Return instance trait (force creation if it does not exist)
|     1: Return existing instance trait (do not create) 
|     0: Return existing instance or class trait (do not create)
|    -1: Return instance trait or force create class trait (i.e. prefix trait) 
|    -2: Return the base trait (after all delegation has been resolved)
+----------------------------------------------------------------------------*/

static PyObject *
_has_traits_trait ( has_traits_object * obj, PyObject * args ) {
 
    has_traits_object * delegate;
    has_traits_object * temp_delegate;
    trait_object      * trait;
    PyObject          * name;
    PyObject          * daname;
    PyObject          * daname2;
	PyObject          * dict;
    int i, instance;
    
    /* Parse arguments, which specify the trait name and whether or not an
       instance specific version of the trait is needed or not: */
	if ( !PyArg_ParseTuple( args, "Oi", &name, &instance ) ) 
        return NULL;
    trait = (trait_object *) get_trait( obj, name, instance );
    if ( (instance >= -1) || (trait == NULL) )
        return (PyObject *) trait;
    
    /* Follow the delegation chain until we find a non-delegated trait: */
    delegate = obj;
    daname   = name;
    Py_INCREF( daname );
    for ( i = 0; ; ) {
        
        if ( trait->delegate_attr_name == NULL ) {
            Py_INCREF( trait );
            Py_DECREF( daname );
            return (PyObject *) trait;
        }
                                  
        dict = delegate->obj_dict;
        if ( (dict != NULL) &&
             ((temp_delegate = (has_traits_object *) PyDict_GetItem( dict, 
                                          trait->delegate_name )) != NULL) ) {
                delegate = temp_delegate;
        } else {
            // fixme: Handle the case when the delegate is not in the 
            //        instance dictionary (could be a method that returns 
            //        the real delegate)
            delegate = (has_traits_object *) has_traits_getattro( delegate, 
                                                       trait->delegate_name );
            if ( delegate == NULL ) 
                break;
            Py_DECREF( delegate );
        }
        // fixme: We need to verify that 'delegate' is of type 'CHasTraits'
        //        before we do the following...
        
        daname2 = trait->delegate_attr_name( trait, obj, daname );
        Py_DECREF( daname );
        daname = daname2;
        if ( ((delegate->itrait_dict == NULL) ||
              ((trait = (trait_object *) dict_getitem( delegate->itrait_dict, 
                      daname )) == NULL)) &&
             ((trait = (trait_object *) dict_getitem( delegate->ctrait_dict, 
                      daname )) == NULL) &&
             ((trait = get_prefix_trait( delegate, daname2 )) == NULL) ) { 
            bad_delegate_error( obj, name );
            break;
        }
            
        if ( trait->ob_type != ctrait_type ) {
            fatal_trait_error();
            break;
        }
        
        if ( ++i >= 100 ) {
            delegation_recursion_error2( obj, name );
            break;
        }
    }
    Py_DECREF( daname );
    return NULL;
}

/*-----------------------------------------------------------------------------
|  Calls notifiers when a trait 'property' is explicitly changed:
+----------------------------------------------------------------------------*/

static PyObject *
_has_traits_property_changed ( has_traits_object * obj, PyObject * args ) {
 
    PyObject     * name;
    PyObject     * old_value;
    PyObject     * new_value;
    trait_object * trait;
    int rc;
    
    /* Parse arguments, which specify the name of the changed trait, the 
       previous value, and the new value: */
	if ( !PyArg_ParseTuple( args, "OOO", &name, &old_value, &new_value ) ) 
        return NULL;
    
    if ( (trait = (trait_object *) get_trait( obj, name, -1 )) == NULL )
        return NULL;
    rc = call_notifiers( trait->notifiers, obj->notifiers, obj, name, 
                         old_value, new_value );
    Py_DECREF( trait );
    if ( rc )
        return NULL;
    Py_INCREF( Py_None );
    return Py_None;
}

/*-----------------------------------------------------------------------------
|  Enables/Disables trait change notification for the object:
+----------------------------------------------------------------------------*/

static PyObject *
_has_traits_change_notify ( has_traits_object * obj, PyObject * args ) {
 
    int enabled;
    
    /* Parse arguments, which specify the new trait notification 
       enabled/disabled state: */
	if ( !PyArg_ParseTuple( args, "i", &enabled ) ) 
        return NULL;
    
    if ( enabled ) {
        obj->flags &= (~HASTRAITS_NO_NOTIFY);
    } else {
        obj->flags |= HASTRAITS_NO_NOTIFY;
    }
    
    Py_INCREF( Py_None );
    return Py_None;
}

/*-----------------------------------------------------------------------------
|  Enables/Disables trait change notifications when this object is assigned to 
|  a trait:
+----------------------------------------------------------------------------*/

static PyObject *
_has_traits_veto_notify ( has_traits_object * obj, PyObject * args ) {
 
    int enabled;
    
    /* Parse arguments, which specify the new trait notification veto 
       enabled/disabled state: */
	if ( !PyArg_ParseTuple( args, "i", &enabled ) ) 
        return NULL;
    
    if ( enabled ) {
        obj->flags |= HASTRAITS_VETO_NOTIFY;
    } else {
        obj->flags &= (~HASTRAITS_VETO_NOTIFY);
    }
    
    Py_INCREF( Py_None );
    return Py_None;
}

/*-----------------------------------------------------------------------------
|  Returns whether or not the object has finished being initialized:
+----------------------------------------------------------------------------*/

static PyObject *
_has_traits_inited ( has_traits_object * obj ) {
 
    if ( obj->flags & HASTRAITS_INITED ) {
        Py_INCREF( Py_True );
        return Py_True;
    }
    Py_INCREF( Py_False );
    return Py_False;
}

/*-----------------------------------------------------------------------------
|  Returns the instance trait dictionary:
+----------------------------------------------------------------------------*/

static PyObject *
_has_traits_instance_traits ( has_traits_object * obj, PyObject * args ) {
    
	if ( !PyArg_ParseTuple( args, "" ) )
        return NULL;
    
    if ( obj->itrait_dict == NULL )
		obj->itrait_dict = (PyDictObject *) PyDict_New();
    Py_XINCREF( obj->itrait_dict );
    return (PyObject *) obj->itrait_dict;
}     

/*-----------------------------------------------------------------------------
|  Returns (and optionally creates) the anytrait 'notifiers' list:
+----------------------------------------------------------------------------*/

static PyObject *
_has_traits_notifiers ( has_traits_object * obj, PyObject * args ) {
 
    PyObject * result;
    PyObject * list;
    int force_create;
    
	if ( !PyArg_ParseTuple( args, "i", &force_create ) )
        return NULL;
    
    result = (PyObject *) obj->notifiers;
    if ( result == NULL ) {
        result = Py_None;
        if ( force_create && ((list = PyList_New( 0 )) != NULL) ) {
            obj->notifiers = (PyListObject *) (result = list);
            Py_INCREF( result );
        }
    }
    Py_INCREF( result );
    return result;
}

/*-----------------------------------------------------------------------------
|  Returns the object's instance dictionary:
+----------------------------------------------------------------------------*/

static PyObject *
get_has_traits_dict ( has_traits_object * obj, void * closure ) {

    PyObject * obj_dict = obj->obj_dict;
    if ( obj_dict == NULL ) {
        obj->obj_dict = obj_dict = PyDict_New();
        if ( obj_dict == NULL )
            return NULL;
    }
    Py_INCREF( obj_dict );
    return obj_dict;
}

/*-----------------------------------------------------------------------------
|  Sets the object's dictionary:
+----------------------------------------------------------------------------*/

static int
set_has_traits_dict ( has_traits_object * obj, PyObject * value, void * closure ) {

    if ( !PyDict_Check( value ) ) 
        return dictionary_error();
    return set_value( &obj->obj_dict, value );
}

/*-----------------------------------------------------------------------------
|  'CHasTraits' instance methods:
+----------------------------------------------------------------------------*/

static PyMethodDef has_traits_methods[] = {
	{ "trait_property_changed", (PyCFunction) _has_traits_property_changed,
      METH_VARARGS,
      PyDoc_STR( "trait_property_changed(name,old_value,new_value)" ) },
	{ "_trait_change_notify", (PyCFunction) _has_traits_change_notify,
      METH_VARARGS,
      PyDoc_STR( "_trait_change_notify(boolean)" ) },
	{ "_trait_veto_notify", (PyCFunction) _has_traits_veto_notify,
      METH_VARARGS,
      PyDoc_STR( "_trait_veto_notify(boolean)" ) },
	{ "traits_inited", (PyCFunction) _has_traits_inited,
      METH_NOARGS,
      PyDoc_STR( "traits_inited()" ) },
	{ "_trait",           (PyCFunction) _has_traits_trait,     METH_VARARGS,
      PyDoc_STR( "_trait(name,instance) -> trait" ) },
	{ "_instance_traits", (PyCFunction) _has_traits_instance_traits,
      METH_VARARGS, 
      PyDoc_STR( "_instance_traits() -> dict" ) },
	{ "_notifiers",       (PyCFunction) _has_traits_notifiers, METH_VARARGS,
      PyDoc_STR( "_notifiers(force_create) -> list" ) },
	{ NULL,	NULL },
};

/*-----------------------------------------------------------------------------
|  'CHasTraits' property definitions:
+----------------------------------------------------------------------------*/

static PyGetSetDef has_traits_properties[] = {
	{ "__dict__", (getter) get_has_traits_dict, (setter) set_has_traits_dict },
	{ 0 }
};

/*-----------------------------------------------------------------------------
|  'CHasTraits' type definition:
+----------------------------------------------------------------------------*/

static PyTypeObject has_traits_type = {
	PyObject_HEAD_INIT( DEFERRED_ADDRESS( &PyType_Type ) )
	0,
	"CHasTraits",
	sizeof( has_traits_object ),
	0,
	(destructor) has_traits_dealloc,                    /* tp_dealloc */
	0,                                                  /* tp_print */
	0,                                                  /* tp_getattr */
	0,                                                  /* tp_setattr */
	0,                                                  /* tp_compare */
	0,                                                  /* tp_repr */
	0,                                                  /* tp_as_number */
	0,                                                  /* tp_as_sequence */
	0,                                                  /* tp_as_mapping */
	0,                                                  /* tp_hash */
	0,                                                  /* tp_call */
	0,                                                  /* tp_str */
	(getattrofunc) has_traits_getattro,                 /* tp_getattro */
	(setattrofunc) has_traits_setattro,                 /* tp_setattro */
	0,					                                /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,/* tp_flags */
	0,                                                  /* tp_doc */
	(traverseproc) has_traits_traverse,                 /* tp_traverse */
	(inquiry) has_traits_clear,                         /* tp_clear */
	0,                                                  /* tp_richcompare */
	0,                                                  /* tp_weaklistoffset */
	0,                                                  /* tp_iter */
	0,                                                  /* tp_iternext */
	has_traits_methods,                                 /* tp_methods */
	0,                                                  /* tp_members */
	has_traits_properties,                              /* tp_getset */
	DEFERRED_ADDRESS( &PyBaseObject_Type ),             /* tp_base */
	0,				        	                            /* tp_dict */
	0,				        	                            /* tp_descr_get */
	0,				        	                            /* tp_descr_set */
	sizeof( has_traits_object ) - sizeof( PyObject * ), /* tp_dictoffset */
	has_traits_init,                                    /* tp_init */
	DEFERRED_ADDRESS( PyType_GenericAlloc ),            /* tp_alloc */
	has_traits_new                                      /* tp_new */
};

/*-----------------------------------------------------------------------------
|  Returns the default value associated with a specified trait:
+----------------------------------------------------------------------------*/

static PyObject *
default_value_for ( trait_object      * trait, 
                    has_traits_object * obj,
                    PyObject          * name ) {
    
    PyObject * result = NULL, * value, * dv, * kw, * tuple;
    
    switch ( trait->default_value_type ) {
        case 0:
        case 1:
            result = trait->default_value;
            Py_INCREF( result );
            break;
        case 2:
            result = (PyObject *) obj;
            Py_INCREF( obj );
            break;
        case 3:
            return PySequence_List( trait->default_value );
        case 4:
            return PyDict_Copy( trait->default_value );
        case 5:
            return call_class( TraitListObject, trait, obj, name, 
                               trait->default_value );
        case 6:
            return call_class( TraitDictObject, trait, obj, name,
                               trait->default_value );
        case 7:
            dv = trait->default_value;
            kw = PyTuple_GET_ITEM( dv, 2 );
            if ( kw == Py_None )
                kw = NULL;
            return PyObject_Call( PyTuple_GET_ITEM( dv, 0 ), 
                                  PyTuple_GET_ITEM( dv, 1 ), kw );
        case 8:
            if ( (tuple = PyTuple_New( 1 )) == NULL )
                return NULL;
            PyTuple_SET_ITEM( tuple, 0, (PyObject *) obj );
            Py_INCREF( obj );
            result = PyObject_Call( trait->default_value, tuple, NULL );
            Py_DECREF( tuple );
            if ( (result != NULL) && (trait->validate != NULL) ) {
                value = trait->validate( trait, obj, name, result );
                Py_DECREF( result );
                return value;
            }
            break;
    }           
    return result;
}    

/*-----------------------------------------------------------------------------
|  Returns the value assigned to a standard Python attribute:
+----------------------------------------------------------------------------*/

static PyObject * 
getattr_python ( trait_object      * trait, 
                 has_traits_object * obj, 
                 PyObject          * name ) {
 
    return PyObject_GenericGetAttr( (PyObject *) obj, name );
}

/*-----------------------------------------------------------------------------
|  Returns the value assigned to a generic Python attribute:
+----------------------------------------------------------------------------*/

static PyObject * 
getattr_generic ( trait_object      * trait, 
                  has_traits_object * obj, 
                  PyObject          * name ) {
                     
    return PyObject_GenericGetAttr( (PyObject *) obj, name );
}                     

/*-----------------------------------------------------------------------------
|  Returns the value assigned to an event trait:
+----------------------------------------------------------------------------*/

static PyObject * 
getattr_event ( trait_object      * trait, 
                has_traits_object * obj, 
                PyObject          * name ) {

    PyErr_Format( PyExc_AttributeError,
        "The %.400s trait of a %.50s instance is an 'event', which is write only.", 
        PyString_AS_STRING( name ), obj->ob_type->tp_name );
    return NULL;            
}

/*-----------------------------------------------------------------------------
|  Returns the value assigned to a standard trait:
+----------------------------------------------------------------------------*/

static PyObject *
getattr_trait ( trait_object      * trait, 
                has_traits_object * obj, 
                PyObject          * name ) {
 
    PyObject * result;
    PyObject * dict = obj->obj_dict;
	if ( dict == NULL ) {
		dict = PyDict_New();
		if ( dict == NULL )
            return NULL;
		obj->obj_dict = dict;
	}
    
	if ( PyString_Check( name ) ) {
        if ( (result = default_value_for( trait, obj, name )) != NULL ) {
            if ( PyDict_SetItem( dict, name, result ) >= 0 )
                return result;
            Py_DECREF( result );
        }
        
        if ( PyErr_ExceptionMatches( PyExc_KeyError ) )
    		PyErr_SetObject( PyExc_AttributeError, name );
        return NULL;
    }
        
#ifdef Py_USING_UNICODE
    if ( PyUnicode_Check( name ) ) {
        name = PyUnicode_AsEncodedString( name, NULL, NULL );
        if ( name == NULL )
		    return NULL;
    } else {
        invalid_attribute_error();
        return NULL;
    }
    
    if ( (result = default_value_for( trait, obj, name )) != NULL ) {
        if ( PyDict_SetItem( dict, name, result ) >= 0 ) {
            Py_DECREF( name );
            return result;
        }
        Py_DECREF( result );
    }
    
    if ( PyErr_ExceptionMatches( PyExc_KeyError ) )
		PyErr_SetObject( PyExc_AttributeError, name );
    Py_DECREF( name );
    return NULL;
#else
    invalid_attribute_error();
    return NULL;
#endif
}

/*-----------------------------------------------------------------------------
|  Returns the value assigned to a delegated trait:
+----------------------------------------------------------------------------*/

static PyObject * 
getattr_delegate ( trait_object      * trait, 
                   has_traits_object * obj, 
                   PyObject          * name ) {

	PyTypeObject * tp;
    PyObject     * delegate_attr_name;
    PyObject     * delegate;
    PyObject     * result;
    PyObject     * dict = obj->obj_dict;
    
    if ( (dict == NULL) || 
         ((delegate = PyDict_GetItem( dict, trait->delegate_name )) == NULL) ){
        // fixme: Handle the case when the delegate is not in the instance
        //        dictionary (could be a method that returns the real delegate)
        delegate = has_traits_getattro( obj, trait->delegate_name );
        if ( delegate == NULL ) 
            return NULL;
        Py_DECREF( delegate );
    }
    
	if ( PyString_Check( name ) ) {
        delegate_attr_name = trait->delegate_attr_name( trait, obj, name );
    	tp = delegate->ob_type;
        
    	if ( tp->tp_getattro != NULL ) {
    		result = (*tp->tp_getattro)( delegate, delegate_attr_name );
            Py_DECREF( delegate_attr_name );
            return result;
        }
        
    	if ( tp->tp_getattr != NULL ) { 
    		result = (*tp->tp_getattr)( delegate, 
                                      PyString_AS_STRING( delegate_attr_name ) );
            Py_DECREF( delegate_attr_name );
            return result;
        }
                                      
    	PyErr_Format( DelegationError,
    	    "The '%.50s' object has no attribute '%.400s' because its %.50s delegate has no attribute '%.400s'.",
    		obj->ob_type->tp_name, PyString_AS_STRING( name ),
            tp->tp_name, PyString_AS_STRING( delegate_attr_name ) );
        Py_DECREF( delegate_attr_name );
    	return NULL;
    }        
        
#ifdef Py_USING_UNICODE
    if ( PyUnicode_Check( name ) ) {
        name = PyUnicode_AsEncodedString( name, NULL, NULL );
        if ( name == NULL )
		    return NULL;
    } else {
        invalid_attribute_error();
        return NULL;
    }

    delegate_attr_name = trait->delegate_attr_name( trait, obj, name );
	tp = delegate->ob_type;
    
	if ( tp->tp_getattro != NULL ) { 
		result = (*tp->tp_getattro)( delegate, delegate_attr_name );
        Py_DECREF( name );
        Py_DECREF( delegate_attr_name );
        return result;
    }
    
	if ( tp->tp_getattr != NULL ) { 
		result = (*tp->tp_getattr)( delegate, 
                                    PyString_AS_STRING( delegate_attr_name ) );
        Py_DECREF( name );
        Py_DECREF( delegate_attr_name );
        return result;
    }                                    
                                  
	PyErr_Format( DelegationError,
	    "The '%.50s' object has no attribute '%.400s' because its %.50s delegate has no attribute '%.400s'.",
		obj->ob_type->tp_name, PyString_AS_STRING( name ),
        tp->tp_name, PyString_AS_STRING( delegate_attr_name ) );
    Py_DECREF( name );
    Py_DECREF( delegate_attr_name );
	return NULL;
#else
    invalid_attribute_error();
    return NULL;
#endif
}

/*-----------------------------------------------------------------------------
|  Raises an exception when a disallowed trait is accessed:
+----------------------------------------------------------------------------*/

static PyObject * 
getattr_disallow ( trait_object      * trait, 
                   has_traits_object * obj, 
                   PyObject          * name ) {
   
    if ( PyString_Check( name ) ) 
        unknown_attribute_error( obj, name );
    else 
        invalid_attribute_error();
    return NULL;                       
}                       

/*-----------------------------------------------------------------------------
|  Returns the value of a constant trait:
+----------------------------------------------------------------------------*/

static PyObject * 
getattr_constant ( trait_object      * trait, 
                   has_traits_object * obj, 
                   PyObject          * name ) {
                       
    Py_INCREF( trait->default_value );
    return trait->default_value;
}                       

/*-----------------------------------------------------------------------------
|  Assigns a value to a specified property trait attribute:
+----------------------------------------------------------------------------*/

static PyObject *
getattr_property0 ( trait_object      * trait, 
                    has_traits_object * obj, 
                    PyObject          * name ) {
    
    return PyObject_Call( trait->delegate_name, empty_tuple, NULL );
}                       

static PyObject *
getattr_property1 ( trait_object      * trait, 
                    has_traits_object * obj, 
                    PyObject          * name ) {
    
    PyObject * result;
    
    PyObject * args = PyTuple_New( 1 );
    if ( args == NULL ) 
        return NULL;
    PyTuple_SET_ITEM( args, 0, (PyObject *) obj );
    Py_INCREF( obj );
    result = PyObject_Call( trait->delegate_name, args, NULL );
    Py_DECREF( args );
    return result;
}                       

static PyObject *
getattr_property2 ( trait_object      * trait, 
                    has_traits_object * obj, 
                    PyObject          * name ) {
    
    PyObject * result;
    
    PyObject * args = PyTuple_New( 2 );
    if ( args == NULL ) 
        return NULL;
    PyTuple_SET_ITEM( args, 0, (PyObject *) obj );
    Py_INCREF( obj );
    PyTuple_SET_ITEM( args, 1, name );
    Py_INCREF( name );
    result = PyObject_Call( trait->delegate_name, args, NULL );
    Py_DECREF( args );
    return result;
}   

static trait_getattr getattr_property_handlers[] = { 
    getattr_property0, getattr_property1, getattr_property2
};    

/*-----------------------------------------------------------------------------
|  Assigns a value to a specified standard Python attribute:
+----------------------------------------------------------------------------*/

static int
setattr_python ( trait_object      * traito, 
                 trait_object      * traitd, 
                 has_traits_object * obj, 
                 PyObject          * name,
                 PyObject          * value ) {
                     
    int rc;
    PyObject * dict = obj->obj_dict;
    
    if ( value != NULL ) {
        if ( dict == NULL ) {
            dict = PyDict_New();
            if ( dict == NULL )
                return -1;
        	obj->obj_dict = dict;
        }
        if ( PyString_Check( name ) ) { 
            if ( PyDict_SetItem( dict, name, value ) >= 0 )
                return 0;
    		if ( PyErr_ExceptionMatches( PyExc_KeyError ) )
    			PyErr_SetObject( PyExc_AttributeError, name );
            return -1;
    	}
#ifdef Py_USING_UNICODE
        if ( PyUnicode_Check( name ) ) {
            name = PyUnicode_AsEncodedString( name, NULL, NULL );
            if ( name == NULL )
        	    return -1;
        } else 
            return invalid_attribute_error();
        rc = PyDict_SetItem( dict, name, value );
        if ( (rc < 0) && PyErr_ExceptionMatches( PyExc_KeyError ) )
             PyErr_SetObject( PyExc_AttributeError, name );
        Py_DECREF( name );
        return rc;
#else   
        return invalid_attribute_error();
#endif
    }

    if ( dict != NULL ) {
        if ( PyString_Check( name ) ) { 
            if ( PyDict_DelItem( dict, name ) >= 0 )
                return 0;
            if ( PyErr_ExceptionMatches( PyExc_KeyError ) )
                unknown_attribute_error( obj, name );
            return -1;
        }
#ifdef Py_USING_UNICODE
        if ( PyUnicode_Check( name ) ) {
            name = PyUnicode_AsEncodedString( name, NULL, NULL );
            if ( name == NULL )
        	    return -1;
        } else 
            return invalid_attribute_error();
            
        rc = PyDict_DelItem( dict, name );
        if ( (rc < 0) && PyErr_ExceptionMatches( PyExc_KeyError ) )
            unknown_attribute_error( obj, name );
        Py_DECREF( name );
        return rc;
#else       
        return invalid_attribute_error();
#endif
    }
    
    if ( PyString_Check( name ) ) {
        unknown_attribute_error( obj, name );
        return -1;
    }
    return invalid_attribute_error();
}

/*-----------------------------------------------------------------------------
|  Assigns a value to a specified generic Python attribute:
+----------------------------------------------------------------------------*/

static int
setattr_generic ( trait_object      * traito, 
                  trait_object      * traitd, 
                  has_traits_object * obj, 
                  PyObject          * name,
                  PyObject          * value ) {
                      
    return PyObject_GenericSetAttr( (PyObject *) obj, name, value );                      
}                      

/*-----------------------------------------------------------------------------
|  Call all notifiers for a specified trait:
+----------------------------------------------------------------------------*/

static int
call_notifiers ( PyListObject      * tnotifiers, 
                 PyListObject      * onotifiers, 
                 has_traits_object * obj, 
                 PyObject          * name,
                 PyObject          * old_value,
                 PyObject          * new_value ) {
                     
    int i, n, new_value_has_traits;
    PyObject * result;
    
    PyObject * args = PyTuple_New( 4 );
    if ( args == NULL )
        return -1;
    
    new_value_has_traits = PyHasTraits_Check( new_value );
    PyTuple_SET_ITEM( args, 0, (PyObject *) obj );
    PyTuple_SET_ITEM( args, 1, name );
    PyTuple_SET_ITEM( args, 2, old_value );
    PyTuple_SET_ITEM( args, 3, new_value );
    Py_INCREF( obj );
    Py_INCREF( name );
    Py_INCREF( old_value );
    Py_INCREF( new_value );
    
    if ( tnotifiers != NULL ) {
        for ( i = 0, n = PyList_GET_SIZE( tnotifiers ); i < n; i++ ) {
            if ( new_value_has_traits && 
                 (((has_traits_object *) new_value)->flags & 
                    HASTRAITS_VETO_NOTIFY) ) {
                Py_DECREF( args );
                return 0;
            }        
            result = PyObject_Call( PyList_GET_ITEM( tnotifiers, i ), 
                                    args, NULL );
            if ( result == NULL ) {
                Py_DECREF( args );
                return -1;
            }
            Py_DECREF( result );
        }
    }
    
    if ( onotifiers != NULL ) {
        for ( i = 0, n = PyList_GET_SIZE( onotifiers ); i < n; i++ ) {
            if ( new_value_has_traits && 
                 (((has_traits_object *) new_value)->flags & 
                    HASTRAITS_VETO_NOTIFY) ) {
                break;
            }        
            result = PyObject_Call( PyList_GET_ITEM( onotifiers, i ), 
                                    args, NULL );
            if ( result == NULL ) {
                Py_DECREF( args );
                return -1;
            }
            Py_DECREF( result );
        }
    }
    
    Py_DECREF( args );
    return 0;
}                     

/*-----------------------------------------------------------------------------
|  Assigns a value to a specified event trait attribute:
+----------------------------------------------------------------------------*/

static int
setattr_event ( trait_object      * traito, 
                trait_object      * traitd, 
                has_traits_object * obj, 
                PyObject          * name,
                PyObject          * value ) {
    
    if ( value != NULL ) {
        if ( traitd->validate != NULL ) {
            value = traitd->validate( traitd, obj, name, value );
            if ( value == NULL ) 
                return -1;
            Py_DECREF( value );
        }
        if ( ((obj->flags & HASTRAITS_NO_NOTIFY) == 0) &&
              ((obj->notifiers != NULL) || (traito->notifiers != NULL)) ) 
            return call_notifiers( traito->notifiers, obj->notifiers, obj, name, 
                                   undefined, value );
    }
    return 0;
}                    

/*-----------------------------------------------------------------------------
|  Assigns a value to a specified normal trait attribute:
+----------------------------------------------------------------------------*/

static int
setattr_trait ( trait_object      * traito, 
                trait_object      * traitd, 
                has_traits_object * obj, 
                PyObject          * name,
                PyObject          * value ) {
            
    int changed;
    int rc;
    PyListObject * tnotifiers = NULL;
    PyListObject * onotifiers = NULL;
    PyObject     * old_value = NULL;
    
    PyObject * dict = obj->obj_dict;
    
    if ( value == NULL ) {
        if ( dict != NULL ) {
            if ( PyString_Check( name ) ) { 
                if ( PyDict_DelItem( dict, name ) >= 0 )
                    return 0;
                if ( !PyErr_ExceptionMatches( PyExc_KeyError ) ) 
                    return -1;
                PyErr_Clear();
                return 0;
            }
#ifdef Py_USING_UNICODE
            if ( PyUnicode_Check( name ) ) {
                name = PyUnicode_AsEncodedString( name, NULL, NULL );
                if ( name == NULL )
            	    return -1;
            } else 
                return invalid_attribute_error();
                
            rc = PyDict_DelItem( dict, name );
            if ( (rc < 0) && PyErr_ExceptionMatches( PyExc_KeyError ) ) {
        		PyErr_Clear();
                rc = 0;
            }
            Py_DECREF( name );
            return rc;
#else       
            return invalid_attribute_error();
#endif
        }
        return 0;
    }
    
    if ( traitd->validate != NULL ) {
        value = traitd->validate( traitd, obj, name, value );
        if ( value == NULL ) 
            return -1;
    } else 
        Py_INCREF( value );
    
    if ( dict == NULL ) {
        obj->obj_dict = dict = PyDict_New();
        if ( dict == NULL ) {
            Py_DECREF( value );
            return -1;
        }
    }
    
    if ( !PyString_Check( name ) ) { 
#ifdef Py_USING_UNICODE
        if ( PyUnicode_Check( name ) ) {
            name = PyUnicode_AsEncodedString( name, NULL, NULL );
            if ( name == NULL ) {
                Py_DECREF( value );
        	    return -1;
            }
        } else {
            Py_DECREF( value );
            return invalid_attribute_error();
        }
#else   
        Py_DECREF( value );
        return invalid_attribute_error();
#endif
    } else
        Py_INCREF( name );
    
    changed   = 0;
    old_value = NULL;
    if ( (obj->flags & HASTRAITS_NO_NOTIFY) == 0 ) {
        tnotifiers = traito->notifiers;
        onotifiers = obj->notifiers;
        if ( (tnotifiers != NULL) || 
             (onotifiers != NULL) || 
             (traitd->post_setattr != NULL) ) {
             old_value = PyDict_GetItem( dict, name );
             if ( old_value == NULL ) {
                 old_value = default_value_for( traitd, obj, name );
                 if ( old_value == NULL ) {
                     Py_DECREF( name );
                     Py_DECREF( value );
                     return -1; 
                 }    
             } else {
                 Py_INCREF( old_value );
             }
             changed = (old_value != value );
             if ( changed &&
                  ((traitd->flags & TRAIT_OBJECT_IDENTITY) == 0) ) {
                 changed = PyObject_RichCompareBool( old_value, value, 
                                                     Py_NE );
                 if ( changed == -1 ) {
                     PyErr_Clear();
                 }
             }
         }
    }
             
    if ( PyDict_SetItem( dict, name, value ) < 0 ) {
        if ( PyErr_ExceptionMatches( PyExc_KeyError ) )
            PyErr_SetObject( PyExc_AttributeError, name );
        Py_XDECREF( old_value );
        Py_DECREF( name );
        Py_DECREF( value );
        return -1; 
    }
     
    rc = 0;
    
    if ( changed ) {
        if ( traitd->post_setattr != NULL ) 
            rc = traitd->post_setattr( traitd, obj, name, value );
        if ( (rc == 0) && ((tnotifiers != NULL) || (onotifiers != NULL)) ) 
            rc = call_notifiers( tnotifiers, onotifiers, obj, name, 
                                 old_value, value );
    }
    Py_XDECREF( old_value );
    Py_DECREF( name );
    Py_DECREF( value );
    return rc;
}                    

/*-----------------------------------------------------------------------------
|  Assigns a value to a specified delegate trait attribute:
+----------------------------------------------------------------------------*/

static int
setattr_delegate ( trait_object      * traito, 
                   trait_object      * traitd, 
                   has_traits_object * obj, 
                   PyObject          * name,
                   PyObject          * value ) {
                       
	PyObject          * dict;
    PyObject          * daname;
    PyObject          * daname2;
    has_traits_object * delegate;
    has_traits_object * temp_delegate;
	int i, result;
    
    /* Follow the delegation chain until we find a non-delegated trait: */
    daname = name;
    Py_INCREF( daname );
    delegate = obj;
    for ( i = 0; ; ) {
        dict = delegate->obj_dict;
        if ( (dict != NULL) && 
             ((temp_delegate = (has_traits_object *) PyDict_GetItem( dict, 
                                          traitd->delegate_name )) != NULL) ) {
            delegate = temp_delegate;
        } else {
            // fixme: Handle the case when the delegate is not in the instance
            //        dictionary (could be a method that returns the real 
            //        delegate)
            delegate = (has_traits_object *) has_traits_getattro( delegate, 
                                                       traitd->delegate_name );
            if ( delegate == NULL ) {
                Py_DECREF( daname );                                               
                return -1;
            }
            Py_DECREF( delegate );
        }
        
        // Verify that 'delegate' is of type 'CHasTraits':
        // fixme: Is there a faster way to do this check?
        if ( !PyHasTraits_Check( delegate ) ) {
            Py_DECREF( daname );
            return bad_delegate_error2( obj, name );
        }
            
        daname2 = traitd->delegate_attr_name( traitd, obj, daname );
        Py_DECREF( daname );
        daname = daname2;
        if ( ((delegate->itrait_dict == NULL) ||
              ((traitd = (trait_object *) dict_getitem( delegate->itrait_dict, 
                      daname )) == NULL)) &&
             ((traitd = (trait_object *) dict_getitem( delegate->ctrait_dict, 
                      daname )) == NULL) &&
             ((traitd = get_prefix_trait( delegate, daname )) == NULL) ) {
            Py_DECREF( daname );
            return bad_delegate_error( obj, name );
        }
            
        if ( traitd->ob_type != ctrait_type ) {
            Py_DECREF( daname );
            return fatal_trait_error();
        }
        
        if ( traitd->delegate_attr_name == NULL ) {
            if ( traito->flags & TRAIT_MODIFY_DELEGATE ) 
                result = setattr_trait( traito, traitd, delegate, daname, 
                                        value );
            else
                result = setattr_trait( traito, traitd, obj, name, value );
            Py_DECREF( daname );
            return result;
        }
        
        if ( ++i >= 100 ) 
            return delegation_recursion_error( obj, name );
    }
} 

/*-----------------------------------------------------------------------------
|  Assigns a value to a specified property trait attribute:
+----------------------------------------------------------------------------*/

static int
setattr_property0 ( trait_object      * traito, 
                    trait_object      * traitd, 
                    has_traits_object * obj, 
                    PyObject          * name,
                    PyObject          * value ) {
                       
    PyObject * result = PyObject_Call( traitd->delegate_prefix, empty_tuple, 
                                       NULL );
    if ( result == NULL ) 
        return -1;
    Py_DECREF( result );
    return 0;
}                       

static int
setattr_property1 ( trait_object      * traito, 
                    trait_object      * traitd, 
                    has_traits_object * obj, 
                    PyObject          * name,
                    PyObject          * value ) {
                       
    PyObject * result;
    
    PyObject * args = PyTuple_New( 1 );
    if ( args == NULL )
        return -1;
    PyTuple_SET_ITEM( args, 0, value );
    Py_INCREF( value );
    result = PyObject_Call( traitd->delegate_prefix, args, NULL );
    Py_DECREF( args );
    if ( result == NULL ) 
        return -1;
    Py_DECREF( result );
    return 0;
}                       

static int
setattr_property2 ( trait_object      * traito, 
                    trait_object      * traitd, 
                    has_traits_object * obj, 
                    PyObject          * name,
                    PyObject          * value ) {
                       
    PyObject * result;
    
    PyObject * args = PyTuple_New( 2 );
    if ( args == NULL )
        return -1;
    PyTuple_SET_ITEM( args, 0, (PyObject *) obj );
    PyTuple_SET_ITEM( args, 1, value );
    Py_INCREF( obj );
    Py_INCREF( value );
    result = PyObject_Call( traitd->delegate_prefix, args, NULL );
    Py_DECREF( args );
    if ( result == NULL ) 
        return -1;
    Py_DECREF( result );
    return 0;
}                       

static int
setattr_property3 ( trait_object      * traito, 
                    trait_object      * traitd, 
                    has_traits_object * obj, 
                    PyObject          * name,
                    PyObject          * value ) {
                       
    PyObject * result;
    
    PyObject * args = PyTuple_New( 3 );
    if ( args == NULL )
        return -1;
    PyTuple_SET_ITEM( args, 0, (PyObject *) obj );
    PyTuple_SET_ITEM( args, 1, name );
    PyTuple_SET_ITEM( args, 2, value );
    Py_INCREF( obj );
    Py_INCREF( name );
    Py_INCREF( value );
    result = PyObject_Call( traitd->delegate_prefix, args, NULL );
    Py_DECREF( args );
    if ( result == NULL ) 
        return -1;
    Py_DECREF( result );
    return 0;
}                       

/*-----------------------------------------------------------------------------
|  Validates then assigns a value to a specified property trait attribute:
+----------------------------------------------------------------------------*/

static int
setattr_validate_property ( trait_object      * traito, 
                            trait_object      * traitd, 
                            has_traits_object * obj, 
                            PyObject          * name,
                            PyObject          * value ) {
                       
    int result;
    
    PyObject * validated = traitd->validate( traitd, obj, name, value );
    if ( validated == NULL ) 
        return -1;
    result = ((trait_setattr) traitd->post_setattr)( traito, traitd, obj, name, 
		                                             validated );
    Py_DECREF( validated );
    return result;
}                       

static PyObject *
setattr_validate0 ( trait_object      * trait, 
                    has_traits_object * obj, 
                    PyObject          * name,
                    PyObject          * value ) {
                       
    return PyObject_Call( trait->py_validate, empty_tuple, NULL );
}                       

static PyObject *
setattr_validate1 ( trait_object      * trait, 
                    has_traits_object * obj, 
                    PyObject          * name,
                    PyObject          * value ) {
                       
    PyObject * validated;
    
    PyObject * args = PyTuple_New( 1 );
    if ( args == NULL )
        return NULL;
    PyTuple_SET_ITEM( args, 0, value );
    Py_INCREF( value );
    validated = PyObject_Call( trait->py_validate, args, NULL );
    Py_DECREF( args );
    return validated;
}                       

static PyObject *
setattr_validate2 ( trait_object      * trait, 
                    has_traits_object * obj, 
                    PyObject          * name,
                    PyObject          * value ) {
                       
    PyObject * validated;
    
    PyObject * args = PyTuple_New( 2 );
    if ( args == NULL )
        return NULL;
    PyTuple_SET_ITEM( args, 0, (PyObject *) obj );
    PyTuple_SET_ITEM( args, 1, value );
    Py_INCREF( obj );
    Py_INCREF( value );
    validated = PyObject_Call( trait->py_validate, args, NULL );
    Py_DECREF( args );
    return validated;
}

static PyObject *
setattr_validate3 ( trait_object      * trait, 
                    has_traits_object * obj, 
                    PyObject          * name,
                    PyObject          * value ) {
                       
    PyObject * validated;
    
    PyObject * args = PyTuple_New( 3 );
    if ( args == NULL )
        return NULL;
    PyTuple_SET_ITEM( args, 0, (PyObject *) obj );
    PyTuple_SET_ITEM( args, 1, name );
    PyTuple_SET_ITEM( args, 2, value );
    Py_INCREF( obj );
    Py_INCREF( name );
    Py_INCREF( value );
    validated = PyObject_Call( trait->py_validate, args, NULL );
    Py_DECREF( args );
    return validated;
}                       

trait_validate setattr_validate_handlers[] = {
    setattr_validate0, setattr_validate1, setattr_validate2, setattr_validate3
};

/*-----------------------------------------------------------------------------
|  Raises an exception when attempting to assign to a disallowed trait:
+----------------------------------------------------------------------------*/

static int
setattr_disallow ( trait_object      * traito, 
                   trait_object      * traitd, 
                   has_traits_object * obj, 
                   PyObject          * name,
                   PyObject          * value ) {
                       
    return set_disallow_error( obj, name );
}                       

/*-----------------------------------------------------------------------------
|  Assigns a value to a specified read-only trait attribute:
+----------------------------------------------------------------------------*/

static int
setattr_readonly ( trait_object      * traito, 
                   trait_object      * traitd, 
                   has_traits_object * obj, 
                   PyObject          * name,
                   PyObject          * value ) {
            
    PyObject * dict;
    PyObject * result;
    
    if ( value == NULL ) 
        return delete_readonly_error( obj, name );
    
    if ( traitd->default_value != undefined )
        return set_readonly_error( obj, name );
    
	dict = obj->obj_dict;
    if ( dict == NULL ) 
        return setattr_python( traito, traitd, obj, name, value );
    
    if ( !PyString_Check( name ) ) { 
#ifdef Py_USING_UNICODE
        if ( PyUnicode_Check( name ) ) {
            name = PyUnicode_AsEncodedString( name, NULL, NULL );
            if ( name == NULL )
        	    return -1;
        } else 
            return invalid_attribute_error();
        
#else   
        return invalid_attribute_error();
#endif
    } else
        Py_INCREF( name );
    
    result = PyDict_GetItem( dict, name );
    Py_DECREF( name );
    if ( (result == NULL) || (result == undefined) )
        return setattr_python( traito, traitd, obj, name, value );
    
    return set_readonly_error( obj, name );
}                    

/*-----------------------------------------------------------------------------
|  Generates exception on attempting to assign to a constant trait:
+----------------------------------------------------------------------------*/

static int
setattr_constant ( trait_object      * traito, 
                   trait_object      * traitd, 
                   has_traits_object * obj, 
                   PyObject          * name,
                   PyObject          * value ) {

    if ( PyString_Check( name ) ) {
	    PyErr_Format( TraitError,
		      "Cannot modify the constant '%.400s' attribute of a '%.50s' object.",
		      PyString_AS_STRING( name ), obj->ob_type->tp_name );
        return -1;
    }
    return invalid_attribute_error();
}                       

/*-----------------------------------------------------------------------------
|  Initializes a CTrait instance:
+----------------------------------------------------------------------------*/

static trait_getattr getattr_handlers[] = {
    getattr_trait,     getattr_python,    getattr_event,  getattr_delegate,    
    getattr_event,     getattr_disallow,  getattr_trait,  getattr_constant,
    getattr_generic,
/*  The following entries are used by the __getstate__ method: */    
    getattr_property0, getattr_property1, getattr_property2,
    NULL
};    

static trait_setattr setattr_handlers[] = {
    setattr_trait,     setattr_python,    setattr_event,     setattr_delegate,    
    setattr_event,     setattr_disallow,  setattr_readonly,  setattr_constant,
    setattr_generic,
/*  The following entries are used by the __getstate__ method: */    
    setattr_property0, setattr_property1, setattr_property2, setattr_property3,
    NULL
};    
    
static int
trait_init ( trait_object * trait, PyObject * args, PyObject * kwds ) {
    
    int kind;
    
	if ( !PyArg_ParseTuple( args, "i", &kind ) )
		return -1;
    
    if ( (kind >= 0) && (kind <= 8) ) {
        trait->getattr = getattr_handlers[ kind ];
        trait->setattr = setattr_handlers[ kind ];
        return 0;
    } 
    return bad_trait_error();
}

/*-----------------------------------------------------------------------------
|  Object clearing method:
+----------------------------------------------------------------------------*/

static int
trait_clear ( trait_object * trait ) {

    Py_CLEAR( trait->default_value );
    Py_CLEAR( trait->py_validate );
    Py_CLEAR( trait->py_post_setattr );
    Py_CLEAR( trait->delegate_name );
    Py_CLEAR( trait->delegate_prefix );
    Py_CLEAR( trait->notifiers );
    Py_CLEAR( trait->handler );
    Py_CLEAR( trait->obj_dict );
    return 0;
}

/*-----------------------------------------------------------------------------
|  Deallocates an unused 'CTrait' instance: 
+----------------------------------------------------------------------------*/

static void 
trait_dealloc ( trait_object * trait ) {
    
    trait_clear( trait );
    trait->ob_type->tp_free( (PyObject *) trait );
}

/*-----------------------------------------------------------------------------
|  Garbage collector traversal method:
+----------------------------------------------------------------------------*/

static int
trait_traverse ( trait_object * trait, visitproc visit, void * arg ) {

    Py_VISIT( trait->default_value );
    Py_VISIT( trait->py_validate );
    Py_VISIT( trait->py_post_setattr );
    Py_VISIT( trait->delegate_name );
    Py_VISIT( trait->delegate_prefix );
    Py_VISIT( (PyObject *) trait->notifiers );
    Py_VISIT( trait->handler );
    Py_VISIT( trait->obj_dict );
	return 0;
}

/*-----------------------------------------------------------------------------
|  Casts a 'CTrait' which attempts to validate the argument passed as being a
|  valid value for the trait:
+----------------------------------------------------------------------------*/

static PyObject *
_trait_cast ( trait_object * trait, PyObject * args ) {
    
    PyObject * obj;
    PyObject * name;
    PyObject * value;
    PyObject * result;
    PyObject * info;
    
    switch ( PyTuple_GET_SIZE( args ) ) {
        case 1: 
            obj   = name = Py_None;
            value = PyTuple_GET_ITEM( args, 0 );
            break;
        case 2: 
            name  = Py_None;
            obj   = PyTuple_GET_ITEM( args, 0 );
            value = PyTuple_GET_ITEM( args, 1 );
            break;
        case 3: 
            obj   = PyTuple_GET_ITEM( args, 0 );
            name  = PyTuple_GET_ITEM( args, 1 );
            value = PyTuple_GET_ITEM( args, 2 );
            break;
        default:
            PyErr_Format( PyExc_TypeError, 
                "Trait cast takes 1, 2 or 3 arguments (%lu given).", 
                PyTuple_GET_SIZE( args ) );
            return NULL;
    }
    if ( trait->validate == NULL ) {
        Py_INCREF( value );
        return value;
    }
	result = trait->validate( trait, (has_traits_object *) obj, name, value );
    if ( result == NULL ) {
        PyErr_Clear();
        info = PyObject_CallMethod( trait->handler, "info", NULL );
        if ( (info != NULL) && PyString_Check( info ) )
            PyErr_Format( PyExc_ValueError, 
                "Invalid value for trait, the value should be %s.", 
                PyString_AS_STRING( info ) );
        else
            PyErr_Format( PyExc_ValueError, "Invalid value for trait." );
        Py_XDECREF( info );
    }
    return result;
}    

/*-----------------------------------------------------------------------------
|  Handles the 'getattr' operation on a 'CHasTraits' instance:
+----------------------------------------------------------------------------*/

static PyObject *
trait_getattro ( trait_object * obj, PyObject * name ) {
    
    PyObject * value = PyObject_GenericGetAttr( (PyObject *) obj, name );
    if ( value != NULL ) 
        return value;
    PyErr_Clear();
    Py_INCREF( Py_None );
    return Py_None;
}

/*-----------------------------------------------------------------------------
|  Sets the value of the 'default_value' field of a CTrait instance:
+----------------------------------------------------------------------------*/

static PyObject *
_trait_default_value ( trait_object * trait, PyObject * args ) {
 
    int        value_type;
    PyObject * value;
    
    if ( PyArg_ParseTuple( args, "" ) ) {
        if ( trait->default_value == NULL ) 
            return Py_BuildValue( "iO", 0, Py_None );
        return Py_BuildValue( "iO", trait->default_value_type, 
                                    trait->default_value );
    }
    if ( !PyArg_ParseTuple( args, "iO", &value_type, &value ) ) 
        return NULL;
    PyErr_Clear();
    if ( (value_type < 0) || (value_type > 8) ) {
        PyErr_Format( PyExc_ValueError, 
                "The default value type must be 0..8, but %d was specified.", 
                value_type );
        return NULL;
    }
    Py_INCREF( value );
    Py_XDECREF( trait->default_value );
    trait->default_value_type = value_type;
    trait->default_value = value;
    Py_INCREF( Py_None );
    return Py_None;
} 

/*-----------------------------------------------------------------------------
|  Calls a Python-based trait validator:
+----------------------------------------------------------------------------*/

static PyObject * 
validate_trait_python ( trait_object * trait, has_traits_object * obj, 
                        PyObject * name, PyObject * value ) {

    PyObject * result;
    
    PyObject * args = PyTuple_New( 3 );
    if ( args == NULL )
        return NULL;
    Py_INCREF( obj );
    Py_INCREF( name );
    Py_INCREF( value );
    PyTuple_SET_ITEM( args, 0, (PyObject *) obj );
    PyTuple_SET_ITEM( args, 1, name );
    PyTuple_SET_ITEM( args, 2, value );
    result = PyObject_Call( trait->py_validate, args, NULL );                            
    Py_DECREF( args );
    return result;
}                            

/*-----------------------------------------------------------------------------
|  Calls the specified validator function:
+----------------------------------------------------------------------------*/

static PyObject * 
call_validator ( PyObject * validator, has_traits_object * obj, 
                 PyObject * name, PyObject * value ) {
          
    PyObject * result;
    
    PyObject * args = PyTuple_New( 3 );
    if ( args == NULL )
        return NULL;
    PyTuple_SET_ITEM( args, 0, (PyObject *) obj );
    PyTuple_SET_ITEM( args, 1, name );
    PyTuple_SET_ITEM( args, 2, value );
    Py_INCREF( obj );
    Py_INCREF( name );
    Py_INCREF( value );
    result = PyObject_Call( validator, args, NULL );
    Py_DECREF( args );
    return result;
}

/*-----------------------------------------------------------------------------
|  Calls the specified type convertor:
+----------------------------------------------------------------------------*/

static PyObject * 
type_converter ( PyObject * type, PyObject * value ) {
          
    PyObject * result;
    
    PyObject * args = PyTuple_New( 1 );
    if ( args == NULL )
        return NULL;
    PyTuple_SET_ITEM( args, 0, value );
    Py_INCREF( value );
    result = PyObject_Call( type, args, NULL );
    Py_DECREF( args );
    return result;
}

/*-----------------------------------------------------------------------------
|  Verifies a Python value is of a specified type (or None):
+----------------------------------------------------------------------------*/

static PyObject * 
validate_trait_type ( trait_object * trait, has_traits_object * obj, 
                      PyObject * name, PyObject * value ) {

    PyObject * type_info = trait->py_validate;
    int kind = PyTuple_GET_SIZE( type_info );
    
    if ( ((kind == 3) && (value == Py_None)) ||
         PyObject_TypeCheck( value, 
                 (PyTypeObject *) PyTuple_GET_ITEM( type_info, kind - 1 ) ) ) {
        Py_INCREF( value );
        return value;
    }
    return raise_trait_error( trait, obj, name, value );
}                            

/*-----------------------------------------------------------------------------
|  Verifies a Python value is an instance of a specified type (or None):
+----------------------------------------------------------------------------*/

static PyObject * 
validate_trait_instance ( trait_object * trait, has_traits_object * obj, 
                          PyObject * name, PyObject * value ) {

    PyObject * type_info = trait->py_validate;
    int kind = PyTuple_GET_SIZE( type_info );
    
    if ( ((kind == 3) && (value == Py_None)) ||
        (PyObject_IsInstance( value, 
             PyTuple_GET_ITEM( type_info, kind - 1 ) ) > 0) ) {
        Py_INCREF( value );
        return value;
    }
    return raise_trait_error( trait, obj, name, value );
}                            

/*-----------------------------------------------------------------------------
|  Verifies a Python value is of a the same type as the object being assigned 
|  to (or None):
+----------------------------------------------------------------------------*/

static PyObject * 
validate_trait_self_type ( trait_object * trait, has_traits_object * obj, 
                           PyObject * name, PyObject * value ) {

    if ( ((PyTuple_GET_SIZE( trait->py_validate ) == 2) && 
          (value == Py_None)) ||
          PyObject_TypeCheck( value, obj->ob_type ) ) { 
        Py_INCREF( value );
        return value;
    }
    return raise_trait_error( trait, obj, name, value );
}                            

/*-----------------------------------------------------------------------------
|  Verifies a Python value is an int within a specified range:
+----------------------------------------------------------------------------*/

static PyObject * 
validate_trait_int ( trait_object * trait, has_traits_object * obj, 
                     PyObject * name, PyObject * value ) {

    register PyObject * low;
    register PyObject * high;
    long exclude_mask;
    long int_value;
    
    PyObject * type_info = trait->py_validate;
    
    if ( PyInt_Check( value ) ) {
        int_value    = PyInt_AS_LONG( value );
        low          = PyTuple_GET_ITEM( type_info, 1 );
        high         = PyTuple_GET_ITEM( type_info, 2 );
        exclude_mask = PyInt_AS_LONG( PyTuple_GET_ITEM( type_info, 3 ) );
        if ( low != Py_None ) {
            if ( (exclude_mask & 1) != 0 ) {
                if ( int_value <= PyInt_AS_LONG( low ) )
                    goto error;
            } else {
                if ( int_value < PyInt_AS_LONG( low ) )
                    goto error;
            }
        }
        if ( high != Py_None ) {
            if ( (exclude_mask & 2) != 0 ) {
                if ( int_value >= PyInt_AS_LONG( high ) )
                    goto error;
            } else {
                if ( int_value > PyInt_AS_LONG( high ) )
                    goto error;
            }
        }
        Py_INCREF( value );
        return value;
    }
error:
    return raise_trait_error( trait, obj, name, value );
}                            

/*-----------------------------------------------------------------------------
|  Verifies a Python value is a float within a specified range:
+----------------------------------------------------------------------------*/

static PyObject * 
validate_trait_float ( trait_object * trait, has_traits_object * obj, 
                       PyObject * name, PyObject * value ) {

    register PyObject * low;
    register PyObject * high;
    long exclude_mask;
    double float_value;
    
    PyObject * type_info = trait->py_validate;
    
    if ( !PyFloat_Check( value ) ) {
        if ( !PyInt_Check( value ) )
            goto error;
        float_value = (double) PyInt_AS_LONG( value );
        value       = PyFloat_FromDouble( float_value );
        if ( value == NULL ) 
            goto error;
        Py_INCREF( value );
    } else {
        float_value = PyFloat_AS_DOUBLE( value );
    }
    low          = PyTuple_GET_ITEM( type_info, 1 );
    high         = PyTuple_GET_ITEM( type_info, 2 );
    exclude_mask = PyInt_AS_LONG( PyTuple_GET_ITEM( type_info, 3 ) );
    if ( low != Py_None ) {
        if ( (exclude_mask & 1) != 0 ) {
            if ( float_value <= PyFloat_AS_DOUBLE( low ) )
                goto error;
        } else {
            if ( float_value < PyFloat_AS_DOUBLE( low ) )
                goto error;
        }
    }
    if ( high != Py_None ) {
        if ( (exclude_mask & 2) != 0 ) {
            if ( float_value >= PyFloat_AS_DOUBLE( high ) )
                goto error;
        } else {
            if ( float_value > PyFloat_AS_DOUBLE( high ) )
                goto error;
        }
    }
    Py_INCREF( value );
    return value;
error:
    return raise_trait_error( trait, obj, name, value );
}                            

/*-----------------------------------------------------------------------------
|  Verifies a Python value is in a specified enumeration:
+----------------------------------------------------------------------------*/

static PyObject * 
validate_trait_enum ( trait_object * trait, has_traits_object * obj, 
                      PyObject * name, PyObject * value ) {

    PyObject * type_info = trait->py_validate;
    if ( PySequence_Contains( PyTuple_GET_ITEM( type_info, 1 ), value ) > 0 ) { 
        Py_INCREF( value );
        return value;
    }
    return raise_trait_error( trait, obj, name, value );
}                            

/*-----------------------------------------------------------------------------
|  Verifies a Python value is in a specified map (i.e. dictionary):
+----------------------------------------------------------------------------*/

static PyObject * 
validate_trait_map ( trait_object * trait, has_traits_object * obj, 
                     PyObject * name, PyObject * value ) {

    PyObject * type_info = trait->py_validate;
    if ( PyDict_GetItem( PyTuple_GET_ITEM( type_info, 1 ), value ) != NULL ) { 
        Py_INCREF( value );
        return value;
    }
    return raise_trait_error( trait, obj, name, value );
}                            

/*-----------------------------------------------------------------------------
|  Verifies a Python value is in a specified prefix map (i.e. dictionary):
+----------------------------------------------------------------------------*/

static PyObject * 
validate_trait_prefix_map ( trait_object * trait, has_traits_object * obj, 
                            PyObject * name, PyObject * value ) {

    PyObject * type_info    = trait->py_validate;
    PyObject * mapped_value = PyDict_GetItem( PyTuple_GET_ITEM( type_info, 1 ), 
                                              value );
    if ( mapped_value != NULL ) { 
        Py_INCREF( mapped_value );
        return mapped_value;
    }
    return call_validator( PyTuple_GET_ITEM( trait->py_validate, 2 ),
                           obj, name, value );
}                            

/*-----------------------------------------------------------------------------
|  Verifies a Python value is a tuple of a specified type and content:
+----------------------------------------------------------------------------*/

static PyObject * 
validate_trait_tuple_check ( PyObject * traits, has_traits_object * obj, 
                             PyObject * name, PyObject * value ) {

    trait_object * itrait;
    PyObject     * bitem, * aitem, * tuple;
    int i, j, n;
    
    if ( PyTuple_Check( value ) ) {
        n = PyTuple_GET_SIZE( traits );
        if ( n == PyTuple_GET_SIZE( value ) ) {
            tuple = NULL;
            for ( i = 0; i < n; i++ ) {
                bitem  = PyTuple_GET_ITEM( value, i );
                itrait = (trait_object *) PyTuple_GET_ITEM( traits, i );
                if ( itrait->validate == NULL ) {
                    aitem = bitem;
                    Py_INCREF( aitem );
                } else
                    aitem = itrait->validate( itrait, obj, name, bitem );
                if ( aitem == NULL ) {
                    PyErr_Clear();
                    Py_XDECREF( tuple );
                    return NULL;
                }
                if ( tuple != NULL ) 
                    PyTuple_SET_ITEM( tuple, i, aitem );
                else if ( aitem != bitem ) {
                    tuple = PyTuple_New( n );
                    if ( tuple == NULL )
                        return NULL;
                    for ( j = 0; j < i; j++ ) {
                        bitem = PyTuple_GET_ITEM( value, j );
                        Py_INCREF( bitem );
                        PyTuple_SET_ITEM( tuple, j, bitem );
                    }
                    PyTuple_SET_ITEM( tuple, i, aitem );
                } else
                    Py_DECREF( aitem );
            }
            if ( tuple != NULL ) 
                return tuple;
            Py_INCREF( value );
            return value;
        }
    }
    return NULL;
}                            

static PyObject * 
validate_trait_tuple ( trait_object * trait, has_traits_object * obj, 
                       PyObject * name, PyObject * value ) {

    PyObject * result = validate_trait_tuple_check( 
                            PyTuple_GET_ITEM( trait->py_validate, 1 ), 
                            obj, name, value );
    if ( result != NULL )
        return result;
    return raise_trait_error( trait, obj, name, value );
}                            

/*-----------------------------------------------------------------------------
|  Verifies a Python value is of a specified (possibly coercable) type:
+----------------------------------------------------------------------------*/

static PyObject * 
validate_trait_coerce_type ( trait_object * trait, has_traits_object * obj, 
                             PyObject * name, PyObject * value ) {
     
    int i, n;
    PyObject * type2;
    
    PyObject * type_info = trait->py_validate; 
    PyObject * type      = PyTuple_GET_ITEM( type_info, 1 ); 
    if ( PyObject_TypeCheck( value, (PyTypeObject *) type ) ) {
        Py_INCREF( value );
        return value;
    }
    n = PyTuple_GET_SIZE( type_info );
    for ( i = 2; i < n; i++ ) {
        type2 = PyTuple_GET_ITEM( type_info, i ); 
        if ( PyObject_TypeCheck( value, (PyTypeObject *) type2 ) ) 
            return type_converter( type, value );
    }
    return raise_trait_error( trait, obj, name, value );
}                            

/*-----------------------------------------------------------------------------
|  Verifies a Python value is of a specified (possibly castable) type:
+----------------------------------------------------------------------------*/

static PyObject * 
validate_trait_cast_type ( trait_object * trait, has_traits_object * obj, 
                           PyObject * name, PyObject * value ) {
     
    PyObject * result;
    
    PyObject * type_info = trait->py_validate; 
    PyObject * type      = PyTuple_GET_ITEM( type_info, 1 ); 
    if ( PyObject_TypeCheck( value, (PyTypeObject *) type ) ) {
        Py_INCREF( value );
        return value;
    }
    if ( (result = type_converter( type, value )) != NULL )
        return result;
    return raise_trait_error( trait, obj, name, value );
}                            

/*-----------------------------------------------------------------------------
|  Verifies a Python value satisifies a specified function validator:
+----------------------------------------------------------------------------*/

static PyObject * 
validate_trait_function ( trait_object * trait, has_traits_object * obj, 
                          PyObject * name, PyObject * value ) {

    PyObject * result;
    
    result = call_validator( PyTuple_GET_ITEM( trait->py_validate, 1 ),
                             obj, name, value );
    if ( result != NULL )
        return result;
    PyErr_Clear();
    return raise_trait_error( trait, obj, name, value );
} 

/*-----------------------------------------------------------------------------
|  Verifies a Python value satisifies a complex trait definition:
+----------------------------------------------------------------------------*/

static PyObject * 
validate_trait_complex ( trait_object * trait, has_traits_object * obj, 
                         PyObject * name, PyObject * value ) {

    int    i, j, k, kind;
    long   int_value, exclude_mask;
    double float_value;
    PyObject * low, * high, * result, * type_info, * type, * type2;
    
    PyObject * list_type_info = PyTuple_GET_ITEM( trait->py_validate, 1 );
    int n = PyTuple_GET_SIZE( list_type_info );
    for ( i = 0; i < n; i++ ) {
        type_info = PyTuple_GET_ITEM( list_type_info, i );
        switch ( PyInt_AsLong( PyTuple_GET_ITEM( type_info, 0 ) ) ) {
            case 0:  /* Type check: */
                kind = PyTuple_GET_SIZE( type_info );
                if ( ((kind == 3) && (value == Py_None)) ||
                     PyObject_TypeCheck( value, (PyTypeObject *) 
                                    PyTuple_GET_ITEM( type_info, kind - 1 ) ) )
                    goto done;
                break;    
            case 1:  /* Instance check: */
                kind = PyTuple_GET_SIZE( type_info );
                if ( ((kind == 3) && (value == Py_None)) ||
                    (PyObject_IsInstance( value, 
                         PyTuple_GET_ITEM( type_info, kind - 1 ) ) > 0) ) 
                    goto done;
                break;    
            case 2:  /* Self type check: */
                if ( ((PyTuple_GET_SIZE( type_info ) == 2) && 
                      (value == Py_None)) ||
                      PyObject_TypeCheck( value, obj->ob_type ) )  
                    goto done;
                break;    
            case 3:  /* Integer range check: */
                if ( PyInt_Check( value ) ) {
                    int_value    = PyInt_AS_LONG( value );
                    low          = PyTuple_GET_ITEM( type_info, 1 );
                    high         = PyTuple_GET_ITEM( type_info, 2 );
                    exclude_mask = PyInt_AS_LONG( 
                                       PyTuple_GET_ITEM( type_info, 3 ) );
                    if ( low != Py_None ) {
                        if ( (exclude_mask & 1) != 0 ) { 
                            if ( int_value <= PyInt_AS_LONG( low  ) )
                                goto error;
                        } else {
                            if ( int_value < PyInt_AS_LONG( low  ) )
                                goto error; 
                        } 
                    }
                    if ( high != Py_None ) {
                        if ( (exclude_mask & 2) != 0 ) {
                            if ( int_value >= PyInt_AS_LONG( high ) )
                                goto error;
                        } else {
                            if ( int_value > PyInt_AS_LONG( high ) )
                                goto error;
                        }
                    }
                    goto done;
                }
                break;
            case 4:  /* Floating point range check: */
                if ( !PyFloat_Check( value ) ) {
                    if ( !PyInt_Check( value ) )
                        break;
                    float_value = (double) PyInt_AS_LONG( value );
                    value       = PyFloat_FromDouble( float_value );
                    if ( value == NULL )
                        break;
                } else {
                    float_value = PyFloat_AS_DOUBLE( value );
                    Py_INCREF( value );
                }
                low          = PyTuple_GET_ITEM( type_info, 1 );
                high         = PyTuple_GET_ITEM( type_info, 2 );
                exclude_mask = PyInt_AS_LONG( 
                                   PyTuple_GET_ITEM( type_info, 3 ) );
                if ( low != Py_None ) {
                    if ( (exclude_mask & 1) != 0 ) {
                        if ( float_value <= PyFloat_AS_DOUBLE( low ) ) 
                            goto error2;
                    } else {
                        if ( float_value < PyFloat_AS_DOUBLE( low ) )
                            goto error2;
                    }
                }
                if ( high != Py_None ) {
                    if ( (exclude_mask & 2) != 0 ) {
                        if ( float_value >= PyFloat_AS_DOUBLE( high ) ) 
                            goto error2;
                    } else {
                        if ( float_value > PyFloat_AS_DOUBLE( high ) )
                            goto error2;
                    }
                }
                goto done2;
            case 5:  /* Enumerated item check: */
                if ( PySequence_Contains( PyTuple_GET_ITEM( type_info, 1 ), 
                                          value ) > 0 )  
                    goto done;
                break;
            case 6:  /* Mapped item check: */
                if ( PyDict_GetItem( PyTuple_GET_ITEM( type_info, 1 ), 
                                     value ) != NULL )  
                    goto done;
                break;
            case 8:  /* Perform 'slow' validate check: */
                return PyObject_CallMethod( PyTuple_GET_ITEM( type_info, 1 ), 
                                  "slow_validate", "(OOO)", obj, name, value );
            case 9:  /* Tuple item check: */
                result = validate_trait_tuple_check( 
                             PyTuple_GET_ITEM( type_info, 1 ), 
                             obj, name, value );
                if ( result != NULL ) 
                    return result;
                break;
            case 10:  /* Prefix map item check: */
                result = PyDict_GetItem( PyTuple_GET_ITEM( type_info, 1 ), 
                                         value );
                if ( result != NULL ) { 
                    Py_INCREF( result );
                    return result;
                }
                result = call_validator( PyTuple_GET_ITEM( type_info, 2 ),
                                         obj, name, value );
                if ( result != NULL )
                    return result;
                PyErr_Clear();
                break;
            case 11:  /* Coercable type check: */
                type = PyTuple_GET_ITEM( type_info, 1 ); 
                if ( PyObject_TypeCheck( value, (PyTypeObject *) type ) ) 
                    goto done;
                k = PyTuple_GET_SIZE( type_info );
                for ( j = 2; j < k; j++ ) {
                    type2 = PyTuple_GET_ITEM( type_info, j ); 
                    if ( PyObject_TypeCheck( value, (PyTypeObject *) type2 ) ) 
                        return type_converter( type, value );
                }
                break;
            case 12:  /* Castable type check */
                type = PyTuple_GET_ITEM( type_info, 1 ); 
                if ( PyObject_TypeCheck( value, (PyTypeObject *) type ) ) 
                    goto done;
                if ( (result = type_converter( type, value )) != NULL )
                    return result;
                PyErr_Clear();
                break;
            case 13:  /* Function validator check: */
                result = call_validator( PyTuple_GET_ITEM( type_info, 1 ), 
                                         obj, name, value );
                if ( result != NULL )
                    return result;
                PyErr_Clear();
                break;
            default:  /* Should never happen...indicates an internal error: */
                goto error;
        }
    }
error:    
    return raise_trait_error( trait, obj, name, value );
error2:
    Py_DECREF( value );
    return raise_trait_error( trait, obj, name, value );
done:    
    Py_INCREF( value );
done2:    
    return value;
}                            

/*-----------------------------------------------------------------------------
|  Sets the value of the 'validate' field of a CTrait instance:
+----------------------------------------------------------------------------*/

static trait_validate validate_handlers[] = {
    validate_trait_type,        validate_trait_instance, 
    validate_trait_self_type,   validate_trait_int,  
    validate_trait_float,       validate_trait_enum,
    validate_trait_map,         validate_trait_complex, 
    NULL,                       validate_trait_tuple,       
    validate_trait_prefix_map,  validate_trait_coerce_type, 
    validate_trait_cast_type,   validate_trait_function,    
    validate_trait_python,
/*  The following entries are used by the __getstate__ method: */    
    setattr_validate0,           setattr_validate1, 
    setattr_validate2,           setattr_validate3
};

static PyObject *
_trait_set_validate ( trait_object * trait, PyObject * args ) {
 
    PyObject * validate;
    PyObject * v1, * v2, * v3;
    int        n, kind;
    
    if ( !PyArg_ParseTuple( args, "O", &validate ) )
        return NULL;
    if ( PyCallable_Check( validate ) ) {
        kind = 14;
        goto done;
    } 
    if ( PyTuple_CheckExact( validate ) ) { 
        n = PyTuple_GET_SIZE( validate );
        if ( n > 0 ) {
            kind = PyInt_AsLong( PyTuple_GET_ITEM( validate, 0 ) );
            switch ( kind ) {
                case 0:  /* Type check: */
                    if ( (n <= 3) && 
                         PyType_Check( PyTuple_GET_ITEM( validate, n - 1 ) ) &&
                         ((n == 2) || 
                          (PyTuple_GET_ITEM( validate, 1 ) == Py_None)) ) 
                        goto done;
                    break;    
                case 1:  /* Instance check: */
                    if ( (n <= 3) && 
                         ((n == 2) || 
                          (PyTuple_GET_ITEM( validate, 1 ) == Py_None)) ) 
                        goto done;
                    break;    
                case 2:  /* Self type check: */
                    if ( (n == 1) ||
                         ((n == 2) && 
                          (PyTuple_GET_ITEM( validate, 1 ) == Py_None)) ) 
                        goto done;
                    break;    
                case 3:  /* Integer range check: */
                    if ( n == 4 ) {
                        v1 = PyTuple_GET_ITEM( validate, 1 );
                        v2 = PyTuple_GET_ITEM( validate, 2 );
                        v3 = PyTuple_GET_ITEM( validate, 3 );
                        if ( ((v1 == Py_None) || PyInt_Check( v1 )) &&
                             ((v2 == Py_None) || PyInt_Check( v2 )) &&
                             PyInt_Check( v3 ) )
                            goto done;
                    }
                    break;
                case 4:  /* Floating point range check: */
                    if ( n == 4 ) {
                        v1 = PyTuple_GET_ITEM( validate, 1 );
                        v2 = PyTuple_GET_ITEM( validate, 2 );
                        v3 = PyTuple_GET_ITEM( validate, 3 );
                        if ( ((v1 == Py_None) || PyFloat_Check( v1 )) &&
                             ((v2 == Py_None) || PyFloat_Check( v2 )) &&
                             PyInt_Check( v3 ) )
                            goto done;
                    }
                    break;
                case 5:  /* Enumerated item check: */
                    if ( n == 2 ) { 
                        v1 = PyTuple_GET_ITEM( validate, 1 );
                        if ( PyTuple_CheckExact( v1 ) ) 
                            goto done;
                    }
                    break;
                case 6:  /* Mapped item check: */
                    if ( n == 2 ) { 
                        v1 = PyTuple_GET_ITEM( validate, 1 );
                        if ( PyDict_Check( v1 ) ) 
                            goto done;
                    }
                    break;
                case 7:  /* TraitComplex item check: */
                    if ( n == 2 ) { 
                        v1 = PyTuple_GET_ITEM( validate, 1 );
                        if ( PyTuple_CheckExact( v1 ) )
                            goto done;
                    }
                    break;
                case 9:  /* TupleOf item check: */
                    if ( n == 2 ) { 
                        v1 = PyTuple_GET_ITEM( validate, 1 );
                        if ( PyTuple_CheckExact( v1 ) )
                            goto done;
                    }
                    break;
                case 10:  /* Prefix map item check: */
                    if ( n == 3 ) { 
                        v1 = PyTuple_GET_ITEM( validate, 1 );
                        if ( PyDict_Check( v1 ) )
                            goto done;
                    }
                    break;
                case 11:  /* Coercable type check: */
                    if ( n >= 2 ) 
                       goto done;
                    break;
                case 12:  /* Castable type check: */
                    if ( n == 2 ) 
                       goto done;
                    break;
                case 13:  /* Function validator check: */
                    if ( n == 2 ) { 
                        v1 = PyTuple_GET_ITEM( validate, 1 );
                        if ( PyCallable_Check( v1 ) )
                            goto done;
                    }
                    break;
            }
		}
    } 
    PyErr_SetString( PyExc_ValueError, 
                     "The argument must be a tuple or callable." );
    return NULL;
done:   
    trait->validate = validate_handlers[ kind ]; 
    Py_INCREF( validate );
    Py_XDECREF( trait->py_validate ); 
    trait->py_validate = validate;
    Py_INCREF( Py_None );
    return Py_None;
}    

/*-----------------------------------------------------------------------------
|  Validates that a particular value can be assigned to an object trait:
+----------------------------------------------------------------------------*/

static PyObject *
_trait_validate ( trait_object * trait, PyObject * args ) {
    
    PyObject * object, * name, * value;
    
    if ( !PyArg_ParseTuple( args, "OOO", &object, &name, &value ) )
        return NULL;
    
    if ( trait->validate == NULL ) {
        Py_INCREF( value );
        return value;
    }
    
    return trait->validate( trait, (has_traits_object *)object, name, value );
}    

/*-----------------------------------------------------------------------------
|  Calls a Python-based trait post_setattr handler:
+----------------------------------------------------------------------------*/

static int 
post_setattr_trait_python ( trait_object * trait, has_traits_object * obj, 
                            PyObject * name, PyObject * value ) {

    PyObject * result;
    
    PyObject * args = PyTuple_New( 3 );
    if ( args == NULL )
        return -1;
    Py_INCREF( obj );
    Py_INCREF( name );
    Py_INCREF( value );
    PyTuple_SET_ITEM( args, 0, (PyObject *) obj );
    PyTuple_SET_ITEM( args, 1, name );
    PyTuple_SET_ITEM( args, 2, value );
    result = PyObject_Call( trait->py_post_setattr, args, NULL );                            
    Py_DECREF( args );
    if ( result == NULL ) 
        return -1;
    Py_DECREF( result );
    return 0;
}   

/*-----------------------------------------------------------------------------
|  Returns the various forms of delegate names:
+----------------------------------------------------------------------------*/

static PyObject *
delegate_attr_name_name ( trait_object      * trait, 
                          has_traits_object * obj,
                          PyObject          * name ) {
              
    Py_INCREF( name );
    return name;
}

static PyObject *
delegate_attr_name_prefix ( trait_object      * trait, 
                            has_traits_object * obj,
                            PyObject          * name ) {
                              
    Py_INCREF( trait->delegate_prefix );
    return trait->delegate_prefix;
}               

static PyObject *
delegate_attr_name_prefix_name ( trait_object      * trait, 
                                 has_traits_object * obj,
                                 PyObject          * name ) {
                                     
    char * p;
    
    int prefix_len    = PyString_GET_SIZE( trait->delegate_prefix );
    int name_len      = PyString_GET_SIZE( name );
    int total_len     = prefix_len + name_len;
    PyObject * result = PyString_FromStringAndSize( NULL, total_len );
    if ( result == NULL ) {
        Py_INCREF( Py_None );
        return Py_None;
    }
    p = PyString_AS_STRING( result );
    memcpy( p, PyString_AS_STRING( trait->delegate_prefix ), prefix_len );
    memcpy( p + prefix_len, PyString_AS_STRING( name ), name_len );
    return result;
}               

static PyObject *
delegate_attr_name_class_name ( trait_object      * trait, 
                                has_traits_object * obj,
                                PyObject          * name ) {
                                     
	PyObject * prefix, * result;
    char     * p;
    int prefix_len, name_len, total_len;
    
	prefix = PyObject_GetAttr( (PyObject *) obj->ob_type, class_prefix );
    // fixme: Should verify that prefix is a string...
	if ( prefix == NULL ) {
		PyErr_Clear();
        Py_INCREF( name );
		return name;
	}
    prefix_len = PyString_GET_SIZE( prefix );
    name_len   = PyString_GET_SIZE( name );
    total_len  = prefix_len + name_len;
    result     = PyString_FromStringAndSize( NULL, total_len );
    if ( result == NULL ) {
        Py_INCREF( Py_None );
        return Py_None;
    }
    p = PyString_AS_STRING( result );
    memcpy( p, PyString_AS_STRING( prefix ), prefix_len );
    memcpy( p + prefix_len, PyString_AS_STRING( name ), name_len );
    Py_DECREF( prefix );
    return result;
}

/*-----------------------------------------------------------------------------
|  Sets the value of the 'post_setattr' field of a CTrait instance:
+----------------------------------------------------------------------------*/

static delegate_attr_name_func delegate_attr_name_handlers[] = {
    delegate_attr_name_name,         delegate_attr_name_prefix,      
    delegate_attr_name_prefix_name,  delegate_attr_name_class_name,  
    NULL
};

static PyObject *
_trait_delegate ( trait_object * trait, PyObject * args ) {
 
    PyObject * delegate_name;
    PyObject * delegate_prefix;
    int prefix_type;
    int modify_delegate;
    
    if ( !PyArg_ParseTuple( args, "O!O!ii", 
                            &PyString_Type, &delegate_name,
                            &PyString_Type, &delegate_prefix, 
                            &prefix_type,   &modify_delegate ) )
        return NULL;
        
    if ( modify_delegate ) {
        trait->flags |= TRAIT_MODIFY_DELEGATE;
    } else {
        trait->flags &= (~TRAIT_MODIFY_DELEGATE);
    }
    trait->delegate_name   = delegate_name;
    trait->delegate_prefix = delegate_prefix;
    Py_INCREF( delegate_name );
    Py_INCREF( delegate_prefix );
    if ( (prefix_type < 0) || (prefix_type > 3) )
        prefix_type = 0;
    trait->delegate_attr_name = delegate_attr_name_handlers[ prefix_type ];
    Py_INCREF( Py_None );
    return Py_None;
}    

/*-----------------------------------------------------------------------------
|  Sets the value of the 'comparison' mode (a 'modify_delegate' alias) of a
|  CTrait instance:
+----------------------------------------------------------------------------*/

static PyObject *
_trait_rich_comparison ( trait_object * trait, PyObject * args ) {
 
    int compare_type;
    
    if ( !PyArg_ParseTuple( args, "i", &compare_type ) ) 
        return NULL;
        
    if ( compare_type == 0 ) {
        trait->flags |= TRAIT_OBJECT_IDENTITY;
    } else {
        trait->flags &= (~TRAIT_OBJECT_IDENTITY);
    }
    Py_INCREF( Py_None );
    return Py_None;
}    

/*-----------------------------------------------------------------------------
|  Sets the 'property' value fields of a CTrait instance:
+----------------------------------------------------------------------------*/

static trait_setattr setattr_property_handlers[] = {
    setattr_property0, setattr_property1, setattr_property2, setattr_property3,
/*  The following entries are used by the __getstate__ method__: */    
    (trait_setattr) post_setattr_trait_python, NULL    
};

static PyObject *
_trait_property ( trait_object * trait, PyObject * args ) {
 
    PyObject * get, * set, * validate, * result, * temp;
    int get_n, set_n, validate_n;

    if ( PyTuple_GET_SIZE( args ) == 0 ) {
        if ( trait->flags & TRAIT_PROPERTY ) {
            result = PyTuple_New( 3 );
            if ( result != NULL ) {
                PyTuple_SET_ITEM( result, 0, temp = trait->delegate_name );
                Py_INCREF( temp );
                PyTuple_SET_ITEM( result, 1, temp = trait->delegate_prefix );
                Py_INCREF( temp );
                PyTuple_SET_ITEM( result, 2, temp = trait->py_validate );
                Py_INCREF( temp );
                Py_INCREF( result );
                return result;
            }
            return NULL;
        } else {
            Py_INCREF( Py_None );
            return Py_None;
        }
    }
    
    if ( !PyArg_ParseTuple( args, "OiOiOi", &get, &get_n, &set, &set_n, 
                                            &validate, &validate_n ) ) 
        return NULL;
    if ( !PyCallable_Check( get ) || !PyCallable_Check( set ) ||
         ((validate != Py_None) && !PyCallable_Check( validate )) ||
         (get_n < 0)      || (get_n > 2) || 
         (set_n < 0)      || (set_n > 3) ||
         (validate_n < 0) || (validate_n > 3) ) {
        PyErr_SetString( PyExc_ValueError, "Invalid arguments." );
        return NULL;
    }
        
    trait->flags  |= TRAIT_PROPERTY;
    trait->getattr = getattr_property_handlers[ get_n ];
	if ( validate != Py_None ) {
        trait->setattr      = setattr_validate_property;
        trait->post_setattr = (trait_post_setattr) setattr_property_handlers[ 
                                                                      set_n ];
        trait->validate     = setattr_validate_handlers[ validate_n ];
	} else
        trait->setattr = setattr_property_handlers[ set_n ];
    trait->delegate_name   = get;
    trait->delegate_prefix = set;
    trait->py_validate     = validate;
    Py_INCREF( get );
    Py_INCREF( set );
    Py_INCREF( validate );
    Py_INCREF( Py_None );
    return Py_None;
}    

/*-----------------------------------------------------------------------------
|  Clones one trait into another:
+----------------------------------------------------------------------------*/

static void 
trait_clone ( trait_object * trait, trait_object * source ) {
    
    trait->flags              = source->flags;
    trait->getattr            = source->getattr;
    trait->setattr            = source->setattr;
    trait->post_setattr       = source->post_setattr;
    trait->py_post_setattr    = source->py_post_setattr;
    trait->validate           = source->validate;
    trait->py_validate        = source->py_validate;
    trait->default_value_type = source->default_value_type;
    trait->default_value      = source->default_value;
    trait->delegate_name      = source->delegate_name;
    trait->delegate_prefix    = source->delegate_prefix;
    trait->delegate_attr_name = source->delegate_attr_name; 
    trait->handler            = source->handler;
    Py_XINCREF( trait->py_post_setattr );
    Py_XINCREF( trait->py_validate );
    Py_XINCREF( trait->delegate_name );
    Py_XINCREF( trait->default_value );
    Py_XINCREF( trait->delegate_prefix );
    Py_XINCREF( trait->handler );
}    

static PyObject *
_trait_clone ( trait_object * trait, PyObject * args ) {
 
    trait_object * source;
    
	if ( !PyArg_ParseTuple( args, "O!", ctrait_type, &source ) )
        return NULL;
    
    trait_clone( trait, source );
                                     
    Py_INCREF( Py_None );
    return Py_None;
}    

/*-----------------------------------------------------------------------------
|  Returns (and optionally creates) the trait 'notifiers' list:
+----------------------------------------------------------------------------*/

static PyObject *
_trait_notifiers ( trait_object * trait, PyObject * args ) {
 
    PyObject * result;
    PyObject * list;
    int force_create;
    
	if ( !PyArg_ParseTuple( args, "i", &force_create ) )
        return NULL;
    
    result = (PyObject *) trait->notifiers;
    if ( result == NULL ) {
        result = Py_None;
        if ( force_create && ((list = PyList_New( 0 )) != NULL) ) {
            trait->notifiers = (PyListObject *) (result = list);
            Py_INCREF( result );
        }
    }
    Py_INCREF( result );
    return result;
}

/*-----------------------------------------------------------------------------
|  Converts a function to an index into a function table:
+----------------------------------------------------------------------------*/

static int
func_index ( void * function, void ** function_table ) {
 
    int i;
    
    for ( i = 0; function != function_table[i]; i++ );
    return i;
}    

/*-----------------------------------------------------------------------------
|  Gets the pickleable state of the trait:
+----------------------------------------------------------------------------*/

static PyObject *
_trait_getstate ( trait_object * trait, PyObject * args ) {
 
    PyObject * result;
    
    if ( !PyArg_ParseTuple( args, "" ) ) 
        return NULL;
    result = PyTuple_New( 15 );
    if ( result == NULL )
        return NULL;
    PyTuple_SET_ITEM( result,  0, PyInt_FromLong( func_index( 
                  (void *) trait->getattr, (void **) getattr_handlers ) ) );
    PyTuple_SET_ITEM( result,  1, PyInt_FromLong( func_index( 
                  (void *) trait->setattr, (void **) setattr_handlers ) ) );
    PyTuple_SET_ITEM( result,  2, PyInt_FromLong( func_index( 
                  (void *) trait->post_setattr, 
                  (void **) setattr_property_handlers ) ) );
    PyTuple_SET_ITEM( result,  3, get_callable_value( trait->py_post_setattr ));
    PyTuple_SET_ITEM( result,  4, PyInt_FromLong( func_index( 
                  (void *) trait->validate, (void **) validate_handlers ) ) );
    PyTuple_SET_ITEM( result,  5, get_callable_value( trait->py_validate ) );
    PyTuple_SET_ITEM( result,  6, PyInt_FromLong( trait->default_value_type ) );
    PyTuple_SET_ITEM( result,  7, get_value( trait->default_value ) );
    PyTuple_SET_ITEM( result,  8, PyInt_FromLong( trait->flags ) );
    PyTuple_SET_ITEM( result,  9, get_value( trait->delegate_name ) );
    PyTuple_SET_ITEM( result, 10, get_value( trait->delegate_prefix ) );
    PyTuple_SET_ITEM( result, 11, PyInt_FromLong( func_index( 
                  (void *) trait->delegate_attr_name, 
                  (void **) delegate_attr_name_handlers ) ) );
    PyTuple_SET_ITEM( result, 12, get_value( NULL ) ); /* trait->notifiers */
    PyTuple_SET_ITEM( result, 13, get_value( trait->handler ) ); 
    PyTuple_SET_ITEM( result, 14, get_value( trait->obj_dict ) ); 
    return result;
} 

/*-----------------------------------------------------------------------------
|  Restores the pickled state of the trait:
+----------------------------------------------------------------------------*/

static PyObject *
_trait_setstate ( trait_object * trait, PyObject * args ) {
 
    PyObject * ignore, * temp, *temp2;
    int getattr_index, setattr_index, post_setattr_index, validate_index, 
        delegate_attr_name_index;
    
    if ( !PyArg_ParseTuple( args, "(iiiOiOiOiOOiOOO)", 
                &getattr_index,             &setattr_index, 
                &post_setattr_index,        &trait->py_post_setattr, 
                &validate_index,            &trait->py_validate,
                &trait->default_value_type, &trait->default_value,
                &trait->flags,              &trait->delegate_name,
                &trait->delegate_prefix,    &delegate_attr_name_index,
                &ignore,                    &trait->handler,
                &trait->obj_dict ) ) 
        return NULL;
        
    trait->getattr      = getattr_handlers[ getattr_index ];
    trait->setattr      = setattr_handlers[ setattr_index ];
    trait->post_setattr = (trait_post_setattr) setattr_property_handlers[ 
                              post_setattr_index ];
    trait->validate     = validate_handlers[ validate_index ];
    trait->delegate_attr_name = delegate_attr_name_handlers[ 
                                    delegate_attr_name_index ];
      
    /* Convert any references to callable methods on the handler back into
       bound methods: */
    temp = trait->py_validate;
    if ( PyInt_Check( temp ) )
        trait->py_validate = PyObject_GetAttrString( trait->handler, 
                                                     "validate" );
    else if ( PyTuple_Check( temp ) &&
              (PyInt_AsLong( PyTuple_GET_ITEM( temp, 0 ) ) == 10) ) {
        temp2 = PyObject_GetAttrString( trait->handler, "validate" );
        Py_INCREF( temp2 );
        Py_DECREF( PyTuple_GET_ITEM( temp, 2 ) );
        PyTuple_SET_ITEM( temp, 2, temp2 );
    }
    if ( PyInt_Check( trait->py_post_setattr ) )
        trait->py_post_setattr = PyObject_GetAttrString( trait->handler, 
                                                         "post_setattr" );
                                    
    Py_INCREF( trait->py_post_setattr );
    Py_INCREF( trait->py_validate );
    Py_INCREF( trait->default_value );
    Py_INCREF( trait->delegate_name );
    Py_INCREF( trait->delegate_prefix );
    Py_INCREF( trait->handler );
    Py_INCREF( trait->obj_dict );
    
    Py_INCREF( Py_None );
    return Py_None;
} 

/*-----------------------------------------------------------------------------
|  Returns the current trait dictionary:
+----------------------------------------------------------------------------*/

static PyObject *
get_trait_dict ( trait_object * trait, void * closure ) {

    PyObject * obj_dict = trait->obj_dict;
    if ( obj_dict == NULL ) {
        trait->obj_dict = obj_dict = PyDict_New();
        if ( obj_dict == NULL )
            return NULL;
    }
    Py_INCREF( obj_dict );
    return obj_dict;
}

/*-----------------------------------------------------------------------------
|  Sets the current trait dictionary:
+----------------------------------------------------------------------------*/

static int
set_trait_dict ( trait_object * trait, PyObject * value, void * closure ) {

    if ( !PyDict_Check( value ) ) 
        return dictionary_error();
    return set_value( &trait->obj_dict, value );
}

/*-----------------------------------------------------------------------------
|  Returns the current trait handler (if any):
+----------------------------------------------------------------------------*/

static PyObject *
get_trait_handler ( trait_object * trait, void * closure ) {

    return get_value( trait->handler );
}

/*-----------------------------------------------------------------------------
|  Sets the current trait dictionary:
+----------------------------------------------------------------------------*/

static int
set_trait_handler ( trait_object * trait, PyObject * value, void * closure ) {

    return set_value( &trait->handler, value );
}

/*-----------------------------------------------------------------------------
|  Returns the current post_setattr (if any):
+----------------------------------------------------------------------------*/

static PyObject *
get_trait_post_setattr ( trait_object * trait, void * closure ) {

    return get_value( trait->py_post_setattr );
}

/*-----------------------------------------------------------------------------
|  Sets the value of the 'post_setattr' field of a CTrait instance:
+----------------------------------------------------------------------------*/

static int
set_trait_post_setattr ( trait_object * trait, PyObject * value, 
                         void * closure ) {
 
    if ( !PyCallable_Check( value ) ) {
        PyErr_SetString( PyExc_ValueError, 
                         "The assigned value must be callable." );
        return -1;
    }
    trait->post_setattr = post_setattr_trait_python;
    return set_value( &trait->py_post_setattr, value );
} 

/*-----------------------------------------------------------------------------
|  'CTrait' instance methods:
+----------------------------------------------------------------------------*/

static PyMethodDef trait_methods[] = {
	{ "__getstate__", (PyCFunction) _trait_getstate,       METH_VARARGS,
	 	PyDoc_STR( "__getstate__()" ) },
	{ "__setstate__", (PyCFunction) _trait_setstate,       METH_VARARGS,
	 	PyDoc_STR( "__setstate__(state)" ) },
	{ "default_value", (PyCFunction) _trait_default_value, METH_VARARGS,
	 	PyDoc_STR( "default_value(default_value)" ) },
	{ "set_validate",  (PyCFunction) _trait_set_validate,  METH_VARARGS,
	 	PyDoc_STR( "set_validate(validate_function)" ) },
	{ "validate",      (PyCFunction) _trait_validate,      METH_VARARGS,
	 	PyDoc_STR( "validate(object,name,value)" ) },
	{ "delegate",      (PyCFunction) _trait_delegate,      METH_VARARGS,
	 	PyDoc_STR( "delegate(delegate_name,prefix,prefix_type,modify_delegate)" ) },
	{ "rich_comparison",  (PyCFunction) _trait_rich_comparison,  METH_VARARGS,
	 	PyDoc_STR( "rich_comparison(comparison_boolean)" ) },
	{ "property",      (PyCFunction) _trait_property,      METH_VARARGS,
	 	PyDoc_STR( "property([get,set,validate])" ) },
	{ "clone",         (PyCFunction) _trait_clone,         METH_VARARGS,
	 	PyDoc_STR( "clone(trait)" ) },
	{ "cast",          (PyCFunction) _trait_cast,          METH_VARARGS,
	 	PyDoc_STR( "cast(value)" ) },
	{ "_notifiers",    (PyCFunction) _trait_notifiers,     METH_VARARGS,
	 	PyDoc_STR( "_notifiers(force_create)" ) },
	{ NULL,	NULL },
};

/*-----------------------------------------------------------------------------
|  'CTrait' property definitions:
+----------------------------------------------------------------------------*/

static PyGetSetDef trait_properties[] = {
	{ "__dict__",     (getter) get_trait_dict,    (setter) set_trait_dict },
	{ "handler",      (getter) get_trait_handler, (setter) set_trait_handler },
	{ "post_setattr", (getter) get_trait_post_setattr, 
                      (setter) set_trait_post_setattr },
	{ 0 }
};

/*-----------------------------------------------------------------------------
|  'CTrait' type definition:
+----------------------------------------------------------------------------*/

static PyTypeObject trait_type = {
	PyObject_HEAD_INIT( DEFERRED_ADDRESS( &PyType_Type ) )
	0,
	"cTrait",
	sizeof( trait_object ),
	0,
	(destructor) trait_dealloc,                    /* tp_dealloc */
	0,                                             /* tp_print */
	0,                                             /* tp_getattr */
	0,                                             /* tp_setattr */
	0,                                             /* tp_compare */
	0,                                             /* tp_repr */
	0,                                             /* tp_as_number */
	0,                                             /* tp_as_sequence */
	0,                                             /* tp_as_mapping */
	0,                                             /* tp_hash */
	0,                                             /* tp_call */
	0,                                             /* tp_str */
	(getattrofunc) trait_getattro,                 /* tp_getattro */
	0,                                             /* tp_setattro */
	0,					                           /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC,/* tp_flags */
	0,                                             /* tp_doc */
	(traverseproc) trait_traverse,                 /* tp_traverse */
	(inquiry) trait_clear,                         /* tp_clear */
	0,                                             /* tp_richcompare */
	0,                                        /* tp_weaklistoffset */
	0,                                             /* tp_iter */
	0,                                             /* tp_iternext */
	trait_methods,                                 /* tp_methods */
	0,                                             /* tp_members */
	trait_properties,                              /* tp_getset */
	DEFERRED_ADDRESS( &PyBaseObject_Type ),        /* tp_base */
    0,                                             /* tp_dict */
    0,                                             /* tp_descr_get */
    0,                                             /* tp_descr_set */
	sizeof( trait_object ) - sizeof( PyObject * ), /* tp_dictoffset */
	(initproc) trait_init,                         /* tp_init */
	DEFERRED_ADDRESS( PyType_GenericAlloc ),       /* tp_alloc */
	DEFERRED_ADDRESS( PyType_GenericNew )          /* tp_new */
};

/*-----------------------------------------------------------------------------
|  Sets the global 'Undefined' and 'Missing' values:
+----------------------------------------------------------------------------*/

static PyObject *
_ctraits_undefined ( PyObject * self, PyObject * args ) {
    
    if ( !PyArg_ParseTuple( args, "O", &undefined ) )
        return NULL;
    Py_INCREF( undefined );
    Py_INCREF( Py_None );
    return Py_None;
}    

/*-----------------------------------------------------------------------------
|  Sets the global 'TraitError' and 'DelegationError' exception types:
+----------------------------------------------------------------------------*/

static PyObject *
_ctraits_exceptions ( PyObject * self, PyObject * args ) {
    
    if ( !PyArg_ParseTuple( args, "OO", &TraitError, &DelegationError ) )
        return NULL;
    Py_INCREF( TraitError );
    Py_INCREF( DelegationError );
    Py_INCREF( Py_None );
    return Py_None;
}    

/*-----------------------------------------------------------------------------
|  Sets the global 'TraitListObject' and 'TraitDictObject' classes:
+----------------------------------------------------------------------------*/

static PyObject *
_ctraits_list_classes ( PyObject * self, PyObject * args ) {
    
    if ( !PyArg_ParseTuple( args, "OO", &TraitListObject, &TraitDictObject ) )
        return NULL;
    Py_INCREF( TraitListObject );
    Py_INCREF( TraitDictObject );
    Py_INCREF( Py_None );
    return Py_None;
}    

/*-----------------------------------------------------------------------------
|  Sets the global 'ctrait_type' class reference:
+----------------------------------------------------------------------------*/

static PyObject *
_ctraits_ctrait ( PyObject * self, PyObject * args ) {
    
    if ( !PyArg_ParseTuple( args, "O", &ctrait_type ) )
        return NULL;
    Py_INCREF( ctrait_type );
    Py_INCREF( Py_None );
    return Py_None;
}    

/*-----------------------------------------------------------------------------
|  'CTrait' instance methods:
+----------------------------------------------------------------------------*/

static PyMethodDef ctraits_methods[] = {
	{ "_undefined",    (PyCFunction) _ctraits_undefined,    METH_VARARGS,
	 	PyDoc_STR( "_undefined(Undefined)" ) },
	{ "_exceptions",   (PyCFunction) _ctraits_exceptions,   METH_VARARGS,
	 	PyDoc_STR( "_exceptions(TraitError,DelegationError)" ) },
	{ "_list_classes", (PyCFunction) _ctraits_list_classes, METH_VARARGS,
	 	PyDoc_STR( "_list_classes(TraitListObject,TraitDictObject)" ) },
	{ "_ctrait",       (PyCFunction) _ctraits_ctrait,       METH_VARARGS,
	 	PyDoc_STR( "_ctrait(CTrait_class)" ) },
	{ NULL,	NULL },
};

/*-----------------------------------------------------------------------------
|  Trait method object definition:
+----------------------------------------------------------------------------*/

typedef struct {
    PyObject_HEAD
    PyObject * tm_name;        /* The name of the method */
    PyObject * tm_func;        /* The callable object implementing the method*/
    PyObject * tm_self;        /* The instance it is bound to, or NULL */
    PyObject * tm_traits;      /* Tuple containing return/arguments traits */
    PyObject * tm_class;       /* The class that asked for the method */
    PyObject * tm_weakreflist; /* List of weak references */
} trait_method_object;

/*-----------------------------------------------------------------------------
|  Instance method objects are used for two purposes:
|  (a) as bound instance methods (returned by instancename.methodname)
|  (b) as unbound methods (returned by ClassName.methodname)
|  In case (b), tm_self is NULL
+----------------------------------------------------------------------------*/

static trait_method_object * free_list;

/*-----------------------------------------------------------------------------
|  Creates a new trait method instance:
+----------------------------------------------------------------------------*/

static PyObject *
create_trait_method ( PyObject * name, PyObject * func, PyObject * self, 
                      PyObject * traits,PyObject * class_obj ) {
    
	register trait_method_object * im;
    
	assert( PyCallable_Check( func ) );
	
	im = free_list;
	if ( im != NULL ) {
		free_list = (trait_method_object *)(im->tm_self);
		PyObject_INIT( im, &trait_method_type );
	} else {
        // fixme: Should we use this form of New if the other 'fixme's are
        // commented out?...
		im = PyObject_GC_New( trait_method_object, &trait_method_type );
		if ( im == NULL )
			return NULL;
	}
	im->tm_weakreflist = NULL;
	Py_INCREF( name );
	im->tm_name = name;
	Py_INCREF( func );
	im->tm_func = func;
	Py_XINCREF( self );
	im->tm_self = self;
	Py_INCREF( traits );
	im->tm_traits = traits;
	Py_XINCREF( class_obj );
	im->tm_class = class_obj;
    // fixme: The following line doesn't link into a separate DLL:
	//_PyObject_GC_TRACK( im );
	return (PyObject *) im;
}

/*-----------------------------------------------------------------------------
|  Gets the value of a trait method attribute:
|
|  The getattr() implementation for trait method objects is similar to
|  PyObject_GenericGetAttr(), but instead of looking in __dict__ it
|  asks tm_self for the attribute.  Then the error handling is a bit
|  different because we want to preserve the exception raised by the
|  delegate, unless we have an alternative from our class. 
+----------------------------------------------------------------------------*/

static PyObject *
trait_method_getattro ( PyObject * obj, PyObject * name ) {
    
	trait_method_object *im = (trait_method_object *) obj;
	PyTypeObject * tp       = obj->ob_type;
	PyObject     * descr    = NULL, * res;
	descrgetfunc f          = NULL;

	if ( PyType_HasFeature( tp, Py_TPFLAGS_HAVE_CLASS ) ) {
		if ( tp->tp_dict == NULL ) {
			if ( PyType_Ready(tp) < 0 )
				return NULL;
		}
		descr = _PyType_Lookup( tp, name );
	}

	f = NULL;
	if ( descr != NULL ) {
		f = TP_DESCR_GET( descr->ob_type );
		if ( (f != NULL) && PyDescr_IsData( descr ) )
			return f( descr, obj, (PyObject *) obj->ob_type );
	}

	res = PyObject_GetAttr( im->tm_func, name );
	if ( (res != NULL) || !PyErr_ExceptionMatches( PyExc_AttributeError ) )
		return res;

	if ( f != NULL ) {
		PyErr_Clear();
		return f( descr, obj, (PyObject *) obj->ob_type );
	}

	if ( descr != NULL ) {
		PyErr_Clear();
		Py_INCREF( descr );
		return descr;
	}

	assert( PyErr_Occurred() );
	return NULL;
}

/*-----------------------------------------------------------------------------
|  Creates a new trait method:
+----------------------------------------------------------------------------*/

static PyObject *
trait_method_new ( PyTypeObject * type, PyObject * args, PyObject * kw ) {
    
	PyObject * name;
	PyObject * func;
    PyObject * traits;

	if ( !PyArg_UnpackTuple( args, "traitmethod", 3, 3, 
                             &name, &func, &traits ) )
		return NULL;
	if ( !PyCallable_Check( func ) ) {
		PyErr_SetString( PyExc_TypeError, "second argument must be callable" );
		return NULL;
	}
    // fixme: Should we sanity check the 'traits' argument here?...
	return create_trait_method( name, func, NULL, traits, NULL );
}

/*-----------------------------------------------------------------------------
|  Deallocates a trait method:
+----------------------------------------------------------------------------*/

static void
trait_method_dealloc ( register trait_method_object * tm ) {
    
    // fixme: The following line complements the _PyObject_GC_TRACK( im )
    // line commented out above...
	//_PyObject_GC_UNTRACK( tm );
	if ( tm->tm_weakreflist != NULL )
		PyObject_ClearWeakRefs( (PyObject *) tm );
	Py_DECREF(  tm->tm_name );
	Py_DECREF(  tm->tm_func );
	Py_XDECREF( tm->tm_self );
	Py_DECREF(  tm->tm_traits );
	Py_XDECREF( tm->tm_class );
	tm->tm_self = (PyObject *) free_list;
	free_list   = tm;
}

/*-----------------------------------------------------------------------------
|  Compare two trait methods:
+----------------------------------------------------------------------------*/

static int
trait_method_compare ( trait_method_object * a, trait_method_object * b ) {
    
	if ( a->tm_self != b->tm_self )
		return (a->tm_self < b->tm_self) ? -1 : 1;
	return PyObject_Compare( a->tm_func, b->tm_func );
}

/*-----------------------------------------------------------------------------
|  Returns the string representation of a trait method:
+----------------------------------------------------------------------------*/

static PyObject *
trait_method_repr ( trait_method_object * a ) {
    
	PyObject * self     = a->tm_self;
	PyObject * func     = a->tm_func;
	PyObject * klass    = a->tm_class;
	PyObject * funcname = NULL, * klassname  = NULL, * result = NULL;
	char     * sfuncname = "?", * sklassname = "?";
 
	funcname = PyObject_GetAttrString( func, "__name__" );
	if ( funcname == NULL ) {
		if ( !PyErr_ExceptionMatches( PyExc_AttributeError ) )
			return NULL;
		PyErr_Clear();
	} else if ( !PyString_Check( funcname ) ) {
		Py_DECREF( funcname );
		funcname = NULL;
	} else
		sfuncname = PyString_AS_STRING( funcname );
        
	if ( klass == NULL )
		klassname = NULL;
	else {
		klassname = PyObject_GetAttrString( klass, "__name__" );
		if (klassname == NULL) {
			if ( !PyErr_ExceptionMatches( PyExc_AttributeError ) )
				return NULL;
			PyErr_Clear();
		} else if ( !PyString_Check( klassname ) ) {
			Py_DECREF( klassname );
			klassname = NULL;
		} else
			sklassname = PyString_AS_STRING( klassname );
	}
    
	if ( self == NULL )
		result = PyString_FromFormat( "<unbound method %s.%s>",
					                  sklassname, sfuncname );
	else {
		/* fixme: Shouldn't use repr() here! */
		PyObject * selfrepr = PyObject_Repr( self );
		if ( selfrepr == NULL )
			goto fail;
		if ( !PyString_Check( selfrepr ) ) {
			Py_DECREF( selfrepr );
			goto fail;
		}
		result = PyString_FromFormat( "<bound method %s.%s of %s>",
					                  sklassname, sfuncname,
					                  PyString_AS_STRING( selfrepr ) );
		Py_DECREF( selfrepr );
	}
    
  fail:
	Py_XDECREF( funcname );
	Py_XDECREF( klassname );
	return result;
}


/*-----------------------------------------------------------------------------
|  Computes the hash of a trait method:
+----------------------------------------------------------------------------*/

static long
trait_method_hash ( trait_method_object * a ) {
    
	long x, y;
	if ( a->tm_self == NULL )
		x = PyObject_Hash( Py_None );
	else
		x = PyObject_Hash( a->tm_self );
	if ( x == -1 )
		return -1;
	y = PyObject_Hash( a->tm_func );
	if ( y == -1 )
		return -1;
	return x ^ y;
}

/*-----------------------------------------------------------------------------
|  Garbage collector traversal method:
+----------------------------------------------------------------------------*/

static int
trait_method_traverse ( trait_method_object * tm, visitproc visit, 
                        void * arg ) {

    Py_VISIT( tm->tm_func );
    Py_VISIT( tm->tm_self );
    Py_VISIT( tm->tm_traits );
    Py_VISIT( tm->tm_class );
	return 0;
}

/*-----------------------------------------------------------------------------
|  Returns the class name of the class:
+----------------------------------------------------------------------------*/

static void
getclassname ( PyObject * class, char * buf, int bufsize ) {
    
	PyObject * name;

	assert( bufsize > 1 );
	strcpy( buf, "?" ); /* Default outcome */
	if ( class == NULL )
		return;
	name = PyObject_GetAttrString( class, "__name__" );
	if ( name == NULL ) {
		/* This function cannot return an exception: */
		PyErr_Clear();
		return;
	}
	if ( PyString_Check( name ) ) {
		strncpy( buf, PyString_AS_STRING( name ), bufsize );
		buf[ bufsize - 1 ] = '\0';
	}
	Py_DECREF( name );
}

/*-----------------------------------------------------------------------------
|  Returns the class name of an instance:
+----------------------------------------------------------------------------*/

static void
getinstclassname ( PyObject * inst, char * buf, int bufsize ) {
    
	PyObject *class;

	if ( inst == NULL ) {
		assert( (bufsize > 0) && ((size_t) bufsize > strlen( "nothing" )) );
		strcpy( buf, "nothing" );
		return;
	}

	class = PyObject_GetAttrString( inst, "__class__" );
	if ( class == NULL ) {
		/* This function cannot return an exception */
		PyErr_Clear();
		class = (PyObject *)(inst->ob_type);
		Py_INCREF( class );
	}
	getclassname( class, buf, bufsize );
	Py_XDECREF( class );
}

/*-----------------------------------------------------------------------------
|  Calls the trait methods and type checks the arguments and result:
+----------------------------------------------------------------------------*/

static PyObject *
trait_method_call ( PyObject * meth, PyObject * arg, PyObject * kw ) {
    
	PyObject     * class,  * result, * self, * new_arg, * func, * value = NULL,
                 * traits, * valid_result, * name = NULL, * dv, * tkw, * tuple; 
    trait_object * trait;
	int from, to, to_args, traits_len, ntraits, ti;
    
    int nargs = PyTuple_GET_SIZE( arg );

    /* Determine if this is an 'unbound' method call: */
    if ( (self = trait_method_GET_SELF( meth )) == NULL ) {
		char clsbuf[256];
		char instbuf[256];
		int  ok;
        
		/* Unbound methods must be called with an instance of the class 
           (or a derived class) as first argument: */
        from = 1;
        class = trait_method_GET_CLASS( meth );
		if ( nargs >= 1 ) {
			self = PyTuple_GET_ITEM( arg, 0 );
            assert( self != NULL );
			ok = PyObject_IsInstance( self, class );
            if ( ok > 0 ) {
                to_args = nargs;
                goto build_args;
            } else if ( ok < 0 )
                return NULL; 
        }
        func = trait_method_GET_FUNCTION( meth );
		getclassname( class, clsbuf, sizeof( clsbuf ) );
		getinstclassname( self, instbuf, sizeof( instbuf ) );
		PyErr_Format( PyExc_TypeError,
			     "unbound method %s%s must be called with "
			     "%s instance as first argument "
			     "(got %s%s instead)",
                 PyString_AS_STRING( trait_method_GET_NAME( meth ) ),
			     PyEval_GetFuncDesc( func ),
			     clsbuf, instbuf, (self == NULL)? "" : " instance" );
		return NULL;
	} 
    from    = 0;
    to_args = nargs + 1;
    
build_args: 
    /* Build the argument list, type checking all arguments as needed: */
    traits     = trait_method_GET_TRAITS( meth );
    traits_len = PyTuple_GET_SIZE( traits );
    ntraits    = traits_len >> 1;
    if ( to_args > ntraits ) 
        return too_may_args_error( trait_method_GET_NAME( meth ), 
                                   ntraits, to_args );
	new_arg = PyTuple_New( ntraits );
	if ( new_arg == NULL )
		return NULL;
	Py_INCREF( self );
	PyTuple_SET_ITEM( new_arg, 0, self );
	for ( to = 1, ti = 3; from < nargs; to++, from++, ti += 2 ) {
		value = PyTuple_GET_ITEM( arg, from );
        assert( value != NULL );
        name  = PyTuple_GET_ITEM( traits, ti );
        trait = (trait_object *) PyTuple_GET_ITEM( traits, ti + 1 );
        if ( kw != NULL ) {
            if ( PyDict_GetItem( kw, name ) != NULL ) {
                Py_DECREF( new_arg );
                return dup_argument_error( trait, meth, from + 1, self, 
                                           name );
            }
        }
        if ( trait->validate == NULL ) {
            Py_INCREF( value );
            PyTuple_SET_ITEM( new_arg, to, value );
            continue;
        }
        value = trait->validate( trait, (has_traits_object *) self, name, 
                                 value );
        if ( value != NULL ) {
            PyTuple_SET_ITEM( new_arg, to, value );
            continue;
        }
        Py_DECREF( new_arg );
        return argument_error( trait, meth, from + 1, self, name, 
                               PyTuple_GET_ITEM( arg, from ) );
	}
    
    /* Substitute default values for any missing arguments: */
    for ( ; ti < traits_len; to++, from++, ti += 2 ) {
        trait = (trait_object *) PyTuple_GET_ITEM( traits, ti + 1 );
        if ( kw != NULL ) {
            name  = PyTuple_GET_ITEM( traits, ti );
            value = PyDict_GetItem( kw, name );
            if ( value != NULL ) {
                if ( trait->validate != NULL ) {
                    valid_result = trait->validate( trait, 
                                     (has_traits_object *) self, name, value );
                    if ( valid_result == NULL ) {
                        Py_DECREF( new_arg );
                        return keyword_argument_error( trait, meth, self, name, 
                                                       value );
                    }
                    value = valid_result;
                } else 
                    Py_INCREF( value );
                PyTuple_SET_ITEM( new_arg, to, value );
                if ( PyDict_DelItem( kw, name ) < 0 ) {
                    Py_DECREF( new_arg );
                    return NULL;
                }
                continue;
            }
        }
        switch ( trait->default_value_type ) {
            case 0:
                value = trait->default_value;
                Py_INCREF( value );
                break;
            case 1:
                Py_DECREF( new_arg );
                return missing_argument_error( trait, meth, from + 1, self,
                                              PyTuple_GET_ITEM( traits, ti ) );
            case 2:
                value = (PyObject *) self;
                Py_INCREF( value );
                break;
            case 3:
            case 5:
                value = PySequence_List( trait->default_value );
                if ( value == NULL ) {
                    Py_DECREF( new_arg );
                    return NULL;
                }
                break;
            case 4:
            case 6:
                value = PyDict_Copy( trait->default_value );
                if ( value == NULL ) {
                    Py_DECREF( new_arg );
                    return NULL;
                }
                break;
            case 7:
                dv  = trait->default_value;
                tkw = PyTuple_GET_ITEM( dv, 2 );
                if ( tkw == Py_None )
                    tkw = NULL;
                value = PyObject_Call( PyTuple_GET_ITEM( dv, 0 ), 
                                       PyTuple_GET_ITEM( dv, 1 ), tkw );
                if ( value == NULL ) {
                    Py_DECREF( new_arg );
                    return NULL;
                }
                break;
            case 8:
                if ( (tuple = PyTuple_New( 1 )) == NULL ) {
                    Py_DECREF( new_arg );
                    return NULL;
                }
                PyTuple_SET_ITEM( tuple, 0, self );
                Py_INCREF( self );
                Py_INCREF( tuple );
                value = PyObject_Call( trait->default_value, tuple, NULL );
                Py_DECREF( tuple );
                if ( value == NULL ) {
                    Py_DECREF( new_arg );
                    return NULL;
                }
                if ( trait->validate != NULL ) {
                    result = trait->validate( trait,
						         (has_traits_object *) self, name, value );
                    Py_DECREF( value );
                    if ( result == NULL ) {
                        Py_DECREF( new_arg );
                        return NULL;
                    }
                    value = result;
                }
                break;
        }
		PyTuple_SET_ITEM( new_arg, to, value );
    }
    
    /* Invoke the method: */
	result = PyObject_Call( trait_method_GET_FUNCTION( meth ), new_arg, kw );
	Py_DECREF( new_arg );
    
    /* Type check the method result (if valid and it was requested): */
    if ( result != NULL ) {
        trait = (trait_object *) PyTuple_GET_ITEM( traits, 0 );
        if ( trait->validate != NULL ) {
            valid_result = trait->validate( trait, (has_traits_object *) self, 
                                            Py_None, result );
            if ( valid_result != NULL ) {
                Py_DECREF( result );
                return valid_result;
            }
            invalid_result_error( trait, meth, self, result );
            Py_DECREF( result );
            return NULL;
        }
    }
    
    /* Finally, return the result: */
	return result;
}

/*-----------------------------------------------------------------------------
|  'get' handler that converts from 'unbound' to 'bound' method:
+----------------------------------------------------------------------------*/

static PyObject *
trait_method_descr_get ( PyObject * meth, PyObject * obj, PyObject * cls ) {
    
	return create_trait_method( trait_method_GET_NAME( meth ), 
	                            trait_method_GET_FUNCTION( meth ), 
                                (obj == Py_None)? NULL: obj, 
                                trait_method_GET_TRAITS( meth ), cls );
}

/*-----------------------------------------------------------------------------
|  Descriptors for trait method attributes:
+----------------------------------------------------------------------------*/

static PyMemberDef trait_method_memberlist[] = {
    { "tm_name",     T_OBJECT,   OFF( tm_name ),    READONLY | RESTRICTED,
      "the name of the method" },
    { "tm_func",     T_OBJECT,   OFF( tm_func ),    READONLY | RESTRICTED,
      "the function (or other callable) implementing a method" },
    { "tm_self",     T_OBJECT,   OFF( tm_self ),    READONLY | RESTRICTED,
      "the instance to which a method is bound; None for unbound methods" },
    { "tm_traits",   T_OBJECT,   OFF( tm_traits ),  READONLY | RESTRICTED,
      "the traits associated with a method" },
    { "tm_class",    T_OBJECT,   OFF( tm_class ),   READONLY | RESTRICTED,
      "the class associated with a method" },
    { NULL }	/* Sentinel */
};

/*-----------------------------------------------------------------------------
|  'CTraitMethod' __doc__ string:
+----------------------------------------------------------------------------*/

PyDoc_STRVAR( trait_method_doc,
"traitmethod(function, traits)\n\
\n\
Create a type checked instance method object.");

/*-----------------------------------------------------------------------------
|  'CTraitMethod' type definition:
+----------------------------------------------------------------------------*/

static PyTypeObject trait_method_type = {
	PyObject_HEAD_INIT( DEFERRED_ADDRESS( &PyType_Type ) )
	0,
	"traitmethod",
	sizeof( trait_method_object ),
	0,
	(destructor) trait_method_dealloc,               /* tp_dealloc */
	0,                                               /* tp_print */
	0,                                               /* tp_getattr */
	0,                                               /* tp_setattr */
	(cmpfunc) trait_method_compare,                  /* tp_compare */
	(reprfunc) trait_method_repr,                    /* tp_repr */
	0,                                               /* tp_as_number */
	0,                                               /* tp_as_sequence */
	0,                                               /* tp_as_mapping */
	(hashfunc) trait_method_hash,                    /* tp_hash */
	trait_method_call,                               /* tp_call */
	0,                                               /* tp_str */
	(getattrofunc) trait_method_getattro,            /* tp_getattro */
    DEFERRED_ADDRESS( PyObject_GenericSetAttr ),     /* tp_setattro */
	0,					                             /* tp_as_buffer */
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,         /* tp_flags */
	trait_method_doc,                                /* tp_doc */
	(traverseproc) trait_method_traverse,            /* tp_traverse */
	0,                                               /* tp_clear */
	0,                                               /* tp_richcompare */
 	offsetof( trait_method_object, tm_weakreflist ), /* tp_weaklistoffset */
	0,				                                 /* tp_iter */
	0,				                                 /* tp_iternext */
	0,				                                 /* tp_methods */
    trait_method_memberlist,                         /* tp_members */
	0,				                                 /* tp_getset */
	0,				                                 /* tp_base */
	0,				                                 /* tp_dict */
	trait_method_descr_get,  	                     /* tp_descr_get */
	0,					                             /* tp_descr_set */
	0,					                             /* tp_dictoffset */
	0,					                             /* tp_init */
	0,					                             /* tp_alloc */
	trait_method_new,		                         /* tp_new */
};

/*-----------------------------------------------------------------------------
|  Performs module and type initialization:
+----------------------------------------------------------------------------*/

PyMODINIT_FUNC
initctraits ( void ) {
    
        PyObject * tmp;

        /* Create the 'ctraits' module: */
	PyObject * module = Py_InitModule3( "ctraits", ctraits_methods, 
                                        ctraits__doc__ );
	if ( module == NULL )
		return;

	/* Create the 'CHasTraits' type: */
	has_traits_type.tp_base  = &PyBaseObject_Type;
	has_traits_type.tp_alloc = PyType_GenericAlloc;
	if ( PyType_Ready( &has_traits_type ) < 0 )
		return;

	Py_INCREF( &has_traits_type );
	if ( PyModule_AddObject( module, "CHasTraits", 
                             (PyObject *) &has_traits_type ) < 0 )
        return;

	/* Create the 'CTrait' type: */
	trait_type.tp_base  = &PyBaseObject_Type;
	trait_type.tp_alloc = PyType_GenericAlloc;
	trait_type.tp_new   = PyType_GenericNew;
	if ( PyType_Ready( &trait_type ) < 0 )
		return;

	Py_INCREF( &trait_type );
	if ( PyModule_AddObject( module, "cTrait", 
                             (PyObject *) &trait_type ) < 0 )
        return;

	/* Create the 'CTraitMethod' type: */
	trait_method_type.tp_base     = &PyBaseObject_Type;
	trait_method_type.tp_setattro = PyObject_GenericSetAttr;
	if ( PyType_Ready( &trait_method_type ) < 0 )
		return;

	Py_INCREF( &trait_method_type );
	if ( PyModule_AddObject( module, "CTraitMethod", 
                             (PyObject *) &trait_method_type ) < 0 )
	return;

	/* Create the 'HasTraitsMonitor' list: */
	tmp = PyList_New( 0 );
	Py_INCREF( tmp );
	if ( PyModule_AddObject( module, "_HasTraits_monitors",
				 (PyObject*) tmp) < 0 ) {
	    return;
	}
    
	_HasTraits_monitors = tmp;

    /* Predefine a Python string == "__class_traits__": */
    class_traits = PyString_FromString( "__class_traits__" );
    
    /* Predefine a Python string == "editor": */
    editor_property = PyString_FromString( "editor" );
    
    /* Predefine a Python string == "__prefix__": */
    class_prefix = PyString_FromString( "__prefix__" );

    /* Create an empty tuple: */    
    empty_tuple = PyTuple_New( 0 );
    
    /* Create the 'is_callable' marker: */
    is_callable = PyInt_FromLong( -1 );
}

