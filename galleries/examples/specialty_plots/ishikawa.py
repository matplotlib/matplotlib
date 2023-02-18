# =============================================================================
# Importing
# =============================================================================
# Python Libraries
from typing import List

import numpy as np

from pathlib import Path 

import matplotlib.pyplot as plt
import matplotlib

# =============================================================================
# AUX FUNCTIONS
# =============================================================================

def drawSpines(data:dict,ax:plt.Axes,parentSpine:None,alpha_ps:float=60.0,alpha_ss:float=0,primary_spines_rel_xpos: List[float] = [np.nan],
               spine_color:str="black",recLevel:int=1,MAX_RECURSION_LEVEL:int=4,DEBUG:bool=False): 
    '''
    Draw an Ishikawa spine with recursion

    Parameters
    ----------
    data : dict
        dictionary structure containing problem, causes...
    ax : plt.Axes
        Matplotlib current axes
    parentSpine : None
        Parent spine object
    alpha_ps : float, optional
        First spine angle using during recursion. The default is 60.0.
    alpha_ss : float, optional
        Secondary spine angle using during recursion. The default is 0.
    primary_spines_rel_xpos : List[float], optional
        Relative X-position of primary spines, a list of float is accepted. The default is [np.nan].
    spine_color : str, optional
        Spine color. The default is "black".
    recLevel : int, optional
        Current recursion level. The default is 1.
    MAX_RECURSION_LEVEL : int, optional
        Maximum recursion level set. The default is 4.
    DEBUG : bool, optional
        Print debug variables. The default is False.

    Raises
    ------
    AttributeError
        Maximum recursion level reached or passed a bad parentSpine object format.

    Returns
    -------
    None.

    '''
    #stop recursion if maximum level reached
    if recLevel > MAX_RECURSION_LEVEL:
        raise AttributeError('Max Recursion Level Reached')
    

    if isinstance(data, dict):
        # switch to correct angle depending on recursion level
        if recLevel % 2 != 0:
            alpha = alpha_ps
        else:
            alpha = alpha_ss
  
        if isinstance(parentSpine,matplotlib.lines.Line2D):
            #calculate parent data
            ([xpb,xpe],[ypb,ype]) = parentSpine.get_data()
        elif isinstance(parentSpine, matplotlib.text.Annotation):
            xpb,ypb = parentSpine.xy
            xpe,ype = parentSpine._x, parentSpine._y#parentSpine._arrow_relpos
        else:
            raise AttributeError('Wrong Spine Graphical Element')
            
        plen = np.sqrt((xpe-xpb)**2+(ype-ypb)**2)
        palpha = np.round(np.degrees(np.arctan2(ype-ypb,xpe-xpb)))  
    
    
        # calculate spacing
        pairs = len(list(data.keys())) // 2 + len(list(data.keys())) % 2 +1 #calculate couple pairs, at least 1 pair to start at middle branch
        spacing = (plen) / pairs
        
        # checking of main spines
        
        
        spine_count=0
        s=spacing
        #draw spine  
        for k in data.keys():
            # calculate arrow position in the graph
            if recLevel==1: #fix primary spines spacing
                if len(primary_spines_rel_xpos)==len(list(data.keys())): #check of list of float is already covered in function def
                    x_end = primary_spines_rel_xpos[spine_count]
                else:
                    x_end = xpe-(s/1.5)*np.cos(np.radians(palpha))
                y_end = ype-s*np.sin(np.radians(palpha))
                
                x_start = x_end -s*np.cos(np.radians(alpha))
                y_start = y_end -s*np.sin(np.radians(alpha))
            else:
                x_end = xpe-s*np.cos(np.radians(palpha))
                y_end = ype-s*np.sin(np.radians(palpha))
                
                x_start = x_end -s*np.cos(np.radians(alpha))
                y_start = y_end -s*np.sin(np.radians(alpha))
            if DEBUG == True:
                print(f'k: {k}, x_start: {x_start}, y_start: {y_start}, x_end: {x_end}, y_end: {y_end}, alpha: {alpha} \n recLevel: {recLevel}, s:{s}, plen:{plen}, palpha:{palpha}')
            
            
            # draw arrow arc
            if recLevel==1:
                props = dict(boxstyle='round', facecolor='lightsteelblue', alpha=1.0)
                spine = ax.annotate(str.upper(k), xy=(x_end, y_end), xytext=(x_start, y_start),arrowprops=dict(arrowstyle="->",facecolor=spine_color),bbox=props,weight='bold')
            else:
                props = dict(boxstyle='round', facecolor='lavender', alpha=1.0)
                spine = ax.annotate(k, xy=(x_end, y_end), xytext=(x_start, y_start),arrowprops=dict(arrowstyle="->",facecolor=spine_color),bbox=props)
            # Call recursion to draw subspines
            drawSpines(data=data[k],ax=ax,
                       parentSpine=spine,alpha_ps=alpha_ps,alpha_ss=alpha_ss,spine_color=spine_color,
                       recLevel=recLevel+1,MAX_RECURSION_LEVEL=MAX_RECURSION_LEVEL) #no primary_spines_rel_xpos is needed to be passed on recursion
    
            # next spine settings - same level
            alpha *= -1
            spine_count += 1
            if  spine_count %2 ==0:
                s= s+spacing 
    return None

def ishikawaplot(data,figSize=(20,10),left_margin:float=0.05,right_margin:float=0.05, alpha_ps:float=60.0,alpha_ss= 0.0, primary_spines_rel_xpos: List[float] = [np.nan],
                 pd_width:int=0.1,spine_color:str="black") -> plt.figure:  
    '''
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    figSize : TYPE, optional
        Matplotlib figure size. The default is (20,10).
    left_margin : float, optional
        Left spacing from frame border. The default is 0.05.
    right_margin : float, optional
        Right spacing from frame border. The default is 0.05.
    alpha_ps : float, optional
        First spine angle using during recursion. The default is 60.0.
    alpha_ss : TYPE, optional
        Secondary spine angle using during recursion. The default is 0.0.
    primary_spines_rel_xpos : list, optional
        Relative X-position of primary spines, a list of float is accepted. The default is [np.nan].
    pd_width : int, optional
        Problem description box relative width. The default is 0.1.
    spine_color : str, optional
        Spine color. The default is "black".

    Returns
    -------
    fig : matplotlib.pyplot.figure
        Figure object containing the Ishikawa plot

    '''
    
    fig = plt.figure(figsize=figSize)
    ax  = fig.gca()
    
    #format axis
    ax.set_xlim(0,1.0)
    ax.set_ylim(0,1.0)    
    ax.axis('off')

    
    #draw main spine
    mainSpine = ax.axhline(y=0.5,xmin=left_margin,xmax=1-right_margin-pd_width,color=spine_color)
    
    #draw fish head
    props = dict(boxstyle='round', facecolor='wheat', alpha=1.0)
    # add problem tag
    ax.text((1-right_margin-pd_width), 0.5, str.upper(list(data.keys())[0]), fontsize=12,weight='bold',verticalalignment='center',horizontalalignment='left', bbox=props)
    
    #draw fish tail
    x = (0.0,0.0,left_margin)
    y = (0.5-left_margin,0.5+left_margin,0.5)
    ax.fill(x,y,color=spine_color)
    
    #draw spines with recursion
    drawSpines(data=data[list(data.keys())[0]],ax=ax,parentSpine=mainSpine,alpha_ps = alpha_ps, alpha_ss = alpha_ss, spine_color=spine_color,primary_spines_rel_xpos=primary_spines_rel_xpos)
    
    
    plt.tight_layout()   
    return fig

# =============================================================================
# Rooting
# =============================================================================
current_dir = Path(__file__).parent
output_path = current_dir / "Ishikawa.jpeg"


# =============================================================================
# USER DATA
# =============================================================================
data = {'problem': {'machine': {'cause1':''},
                    'process': {'cause2': {'subcause1':'','subcause2':''}
                               },
                    'man':     {'cause3': {'subcause3':'','subcause4':''}
                               },
                    'design':  {'cause4': {'subcause5':'','subcause6':''}
                                }
                    
                   }
       }


# =============================================================================
# MAIN CALL
# =============================================================================
if __name__ == '__main__':
    
    #try also without opt primary_spines rel_xpos
    fig = ishikawaplot(data,figSize=(20,10),primary_spines_rel_xpos=[0.8,0.7,0.6,0.4])
    
    fig.show()
    
    fig.savefig(output_path, 
                     #dpi=800, 
                     format=None, 
                     #metadata=None,
                     bbox_inches=None, 
                     pad_inches=0.0,
                     facecolor='auto', 
                     edgecolor='auto',
                     orientation='landscape',
                     transparent=False,
                     backend=None)
    