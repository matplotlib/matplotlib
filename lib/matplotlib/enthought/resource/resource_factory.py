""" Default base-class for resource factories. """


class ResourceFactory:
    """ Default base-class for resource factories. """
    
    ###########################################################################
    # 'ResourceFactory' interface.
    ###########################################################################

    def image_from_file(self, filename):
        """ Creates an image from the data in the specified filename. """
        
        raise NotImplemented

    def image_from_data(self, data):
        """ Creates an image from the specified data. """
        
        raise NotImplemented
    
#### EOF ######################################################################
