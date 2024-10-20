import logging
import time

from PIL import Image, ImageChops
from PIL.ImageGrab import grab

from pyvirtualdisplay import Display

log = logging.getLogger(__name__)


class DisplayTimeoutError(Exception):
    pass


def autocrop(im, bgcolor):
    """Crop borders off an image.

    :param im: Source image.
    :param bgcolor: Background color, using either a color tuple.
    :return: An image without borders, or None if there's no actual content in the image.
    """
    if im.mode != "RGB":
        im = im.convert("RGB")
    bg = Image.new("RGB", im.size, bgcolor)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return None  # no contents


class SmartDisplay(Display):
    def autocrop(self, im):
        """Crop borders off an image.

        :param im: Source image.
        :return: An image without borders, or None if there's no actual content in the image.
        """
        return autocrop(im, self._bgcolor)

    def grab(self, autocrop=True):
        # TODO: use Xvfb fbdir option for screenshot
        img = grab(xdisplay=self.new_display_var)

        if autocrop:
            img = self.autocrop(img)
        return img

    def waitgrab(self, timeout=60, autocrop=True, cb_imgcheck=None):
        """start process and create screenshot.
        Repeat screenshot until it is not empty and
        cb_imgcheck callback function returns True
        for current screenshot.

        :param autocrop: True -> crop screenshot
        :param timeout: int
        :param cb_imgcheck: None or callback for testing img,
                            True = accept img,
                            False = reject img
        """
        t = 0
        sleep_time = 0.3  # for fast windows
        repeat_time = 0.5
        while 1:
            log.debug("sleeping %s secs" % str(sleep_time))
            time.sleep(sleep_time)
            t += sleep_time
            img = self.grab(autocrop=False)
            img_crop = self.autocrop(img)
            if autocrop:
                img = img_crop
            if img_crop:
                if not cb_imgcheck:
                    break
                if cb_imgcheck(img):
                    break
            sleep_time = repeat_time
            repeat_time += 0.5  # progressive
            if t > timeout:
                msg = "Timeout! elapsed time:%s timeout:%s " % (t, timeout)
                raise DisplayTimeoutError(msg)
                # break

            log.debug("screenshot is empty, next try..")
        assert img
        #        if not img:
        #            log.debug('screenshot is empty!')
        return img
