"""
Some io tools for excel -- requires xlwt

Example usage:

    import matplotlib.mlab as mlab
    import mpl_toolkits.exceltools as exceltools

    r = mlab.csv2rec('somefile.csv', checkrows=0)

    formatd = dict(
        weight = mlab.FormatFloat(2),
        change = mlab.FormatPercent(2),
        cost   = mlab.FormatThousands(2),
        )


    exceltools.rec2excel(r, 'test.xls', formatd=formatd)
    mlab.rec2csv(r, 'test.csv', formatd=formatd)

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import copy
import numpy as np

import xlwt as excel

import matplotlib.cbook as cbook
import matplotlib.mlab as mlab


def xlformat_factory(format):
    """
    copy the format, perform any overrides, and attach an xlstyle instance
    copied format is returned
    """

    #if we have created an excel format already using this format,
    #don't recreate it; mlab.FormatObj override has to make objs with
    #the same props hash to the same value
    key = hash(format)
    fmt_ = xlformat_factory.created_formats.get(key)
    if fmt_ is not None:
        return fmt_

    format = copy.deepcopy(format)

    xlstyle = excel.XFStyle()
    if isinstance(format, mlab.FormatPercent):
       zeros = ''.join(['0']*format.precision)
       xlstyle.num_format_str = '0.%s%%;[RED]-0.%s%%'%(zeros, zeros)
       format.scale = 1.
    elif isinstance(format, mlab.FormatFloat):
        if format.precision>0:
            zeros = ''.join(['0']*format.precision)
            xlstyle.num_format_str = '#,##0.%s;[RED]-#,##0.%s'%(zeros, zeros)
        else:
            xlstyle.num_format_str = '#,##;[RED]-#,##'
    elif isinstance(format, mlab.FormatInt):
        xlstyle.num_format_str = '#,##;[RED]-#,##'
    else:
        xlstyle = None

    format.xlstyle = xlstyle

    xlformat_factory.created_formats[ key ] = format

    return format

xlformat_factory.created_formats = {}

def rec2excel(r, ws, formatd=None, rownum=0, colnum=0, nanstr='NaN', infstr='Inf'):
    """
    save record array r to excel xlwt worksheet ws
    starting at rownum.  if ws is string like, assume it is a
    filename and save to it

    start writing at rownum, colnum

    formatd is a dictionary mapping dtype name -> mlab.Format instances

    nanstr is the string that mpl will put into excel for np.nan value
    The next rownum after writing is returned
    """

    autosave = False
    if cbook.is_string_like(ws):
        filename = ws
        wb = excel.Workbook()
        ws = wb.add_sheet('worksheet')
        autosave = True


    if formatd is None:
        formatd = dict()

    formats = []
    font = excel.Font()
    font.bold = True

    stylehdr = excel.XFStyle()
    stylehdr.font = font

    for i, name in enumerate(r.dtype.names):
        dt = r.dtype[name]
        format = formatd.get(name)
        if format is None:
            format = mlab.defaultformatd.get(dt.type, mlab.FormatObj())

        format = xlformat_factory(format)
        ws.write(rownum, colnum+i, name, stylehdr)
        formats.append(format)

    rownum+=1


    ind = np.arange(len(r.dtype.names))
    for row in r:

        for i in ind:
            val = row[i]
            format = formats[i]
            val = format.toval(val)
            if mlab.safe_isnan(val):
                ws.write(rownum, colnum+i, nanstr)
            elif mlab.safe_isinf(val):
                sgn = np.sign(val)
                if sgn>0: s = infstr
                else: s = '-%s'%infstr
                ws.write(rownum, colnum+i, s)
            elif format.xlstyle is None:
                ws.write(rownum, colnum+i, val)
            else:
                ws.write(rownum, colnum+i, val, format.xlstyle)
        rownum += 1

    if autosave:
        wb.save(filename)
    return rownum
