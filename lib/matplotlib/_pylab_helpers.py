import sys, gc

def error_msg(msg):
    print >>sys.stderr, msgs

class Gcf(object):
    _activeQue = []
    figs = {}

    def get_fig_manager(num):
        figManager = Gcf.figs.get(num, None)
        if figManager is not None: Gcf.set_active(figManager)
        return figManager
    get_fig_manager = staticmethod(get_fig_manager)

    def destroy(num):

        if not Gcf.has_fignum(num): return
        figManager = Gcf.figs[num]

        oldQue = Gcf._activeQue[:]
        Gcf._activeQue = []
        for f in oldQue:
            if f != figManager: Gcf._activeQue.append(f)

        del Gcf.figs[num]
        #print len(Gcf.figs.keys()), len(Gcf._activeQue)
        figManager.destroy()
        gc.collect()

    destroy = staticmethod(destroy)

    def has_fignum(num):
        return num in Gcf.figs
    has_fignum = staticmethod(has_fignum)

    def get_all_fig_managers():
        return Gcf.figs.values()
    get_all_fig_managers = staticmethod(get_all_fig_managers)

    def get_num_fig_managers():
        return len(Gcf.figs.values())
    get_num_fig_managers = staticmethod(get_num_fig_managers)


    def get_active():
        if len(Gcf._activeQue)==0:
            return None
        else: return Gcf._activeQue[-1]
    get_active = staticmethod(get_active)

    def set_active(manager):
        oldQue = Gcf._activeQue[:]
        Gcf._activeQue = []
        for m in oldQue:
            if m != manager: Gcf._activeQue.append(m)
        Gcf._activeQue.append(manager)
        Gcf.figs[manager.num] = manager
    set_active = staticmethod(set_active)
