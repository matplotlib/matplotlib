""" Resource references. """


# Enthought library imports.
from matplotlib.enthought.traits import Any, HasTraits, Instance

# Local imports.
from resource_factory import ResourceFactory


class ResourceReference(HasTraits):
    """ Abstract base class for resource references.

    Resource references are returned from calls to 'locate_reference' on the
    resource manager.

    """

    # The resource factory that will be used to load the resource.
    resource_factory = Instance(ResourceFactory) # ReadOnly
    
    ###########################################################################
    # 'ResourceReference' interface.
    ###########################################################################

    def load(self):
        """ Loads the resource. """

        raise NotImplementedError


class ImageReference(ResourceReference):
    """ A reference to an image resource. """

    # Iff the image was found in a file then this is the name of that file.
    filename = Any # ReadOnly

    # Iff the image was found in a zip file then this is the image data that
    # was read from the zip file.
    data = Any # ReadOnly

    
    def __init__(self, resource_factory, filename=None, data=None):
        """ Creates a new image reference. """

        self.resource_factory = resource_factory
        self.filename = filename
        self.data = data
        
        return

    ###########################################################################
    # 'ResourceReference' interface.
    ###########################################################################

    def load(self):
        """ Loads the resource. """

        if self.filename is not None:
            image = self.resource_factory.image_from_file(self.filename)

        elif self.data is not None:
            image = self.resource_factory.image_from_data(self.data)

        else:
            raise ValueError("Image reference has no filename OR data")

        return image
    
#### EOF ######################################################################
