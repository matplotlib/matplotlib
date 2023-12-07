"Start Xvfb server and open xmessage window. Thread safe."

import threading

from easyprocess import EasyProcess

from pyvirtualdisplay.smartdisplay import SmartDisplay


def thread_function(index):
    # manage_global_env=False is thread safe
    with SmartDisplay(manage_global_env=False) as disp:
        cmd = ["xmessage", str(index)]
        # disp.new_display_var should be used for new processes
        # disp.env() copies global os.environ and adds disp.new_display_var
        with EasyProcess(cmd, env=disp.env()):
            img = disp.waitgrab()
            img.save("xmessage{}.png".format(index))


t1 = threading.Thread(target=thread_function, args=(1,))
t2 = threading.Thread(target=thread_function, args=(2,))
t1.start()
t2.start()
t1.join()
t2.join()
