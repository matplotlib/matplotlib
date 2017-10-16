from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from _tkinter.tklib_cffi import ffi as tkffi, lib as tklib
from _tkinter.tclobj import FromTclStringNDArray
import numpy as np

app = None

def PyAggImagePhoto(photoimage, data_as_str, mode, bbox_as_str=None):
    interp = PyAggImagePhoto.interp
    if not tklib.Tk_MainWindow(interp):
        raise _tkinter.TclError("this isn't a Tk application")
    photo = tklib.Tk_FindPhoto(interp, photoimage)
    if not photo:
        tklib.Tcl_AppendResult(interp, "destination photo must exist", 0)
        return tklib.TCL_ERROR
    data = FromTclStringNDArray(data_as_str)
    if bbox_as_str:
        try:
            bbox = FromTclStringNDArray(bbox_as_str)
        except:
            tklib.Tcl_AppendResult(interp, "bbox not valid", 0)
            return tklib.TCL_ERROR
        destx = int(bbox[0, 0])
        desty = data.shape[0] - int(bbox[1, 1])
        destwidth = int(bbox[1, 0] - bbox[0, 0])
        destheight = int(bbox[1, 1] - bbox[0, 1])
        deststride = 4 * destwidth;
        destbuffer = np.empty([destheight, destwidth, 4], dtype='uint8')
        has_bbox = True
        destbuffer[:,:,:] = data[desty:desty+destheight, destx:destx+destwidth,:]
    else:
        has_bbox = False
        destbuffer = None
        destx = desty = destwidth = destheight = deststride = 0;
    pBlock = tkffi.new('Tk_PhotoImageBlock[1]')
    block = pBlock[0]
    block.pixelSize = 1
    if mode == 0:
        block.offset[0] = block.offset[1] = block.offset[2] = 0
        nval = 1
    else:
        block.offset[0] = 0
        block.offset[1] = 1
        block.offset[2] = 2
        if mode == 1:
            block.offset[3] = 0
            block.pixelSize = 3
            nval = 3
        else:
            block.offset[3] = 3
            block.pixelSize = 4
    if has_bbox:
        block.width = destwidth
        block.height = destheight
        block.pitch = deststride
        block.pixelPtr = tkffi.from_buffer(destbuffer)

        tklib.Tk_PhotoPutBlock_NoComposite(photo, pBlock, destx, desty,
                destwidth, destheight);

    else:
        block.width = data.shape[1]
        block.height = data.shape[0]
        block.pitch = data.strides[0]
        block.pixelPtr = tkffi.from_buffer(data)

        #/* Clear current contents */
        tklib.Tk_PhotoBlank(photo);
        #/* Copy opaque block to photo image, and leave the rest to TK */
        tklib.Tk_PhotoPutBlock_NoComposite(photo, pBlock, 0, 0,
                 block.width, block.height);
    return tklib.TCL_OK

def tkinit(tk):
    tk.createcommand(b"PyAggImagePhoto", PyAggImagePhoto)
    PyAggImagePhoto.interp = tk.interp

