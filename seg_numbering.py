# Created as test harness for 'Axes.setSegmentMaskNumbers(imgSegmentMask, **kwargs)'
"""
Algorithm used in populating the segment mid-point with the segment number
created by 'skimage.segmentation.quickshift(...)':
--------------------------------------------------

xLeft = 0.14
xRight = 0.8
yTop = 0.95
yBottom = 0.06

for num in sortedSegmentNums:
    (midX, midY) = self.getSegmentMidpoint(imgSegmentMask, num)

    xFigText = ((xRight - xLeft) * (midX / endX)) + xLeft
    yFigText = (-(yTop - yBottom) * (midY / endY)) + yTop

    if (num not in manual_segments):
        plt.figtext(xFigText, yFigText, num)
"""

import matplotlib
import matplotlib.pyplot as plt
import skimage.segmentation


def displayImgSegments(imgSegmentMask):
    fig, ax = plt.subplots()

    # Add awkward shaped segments manually
    kwargs = {'segment_values': [
              {'num': 10, 'x': 0.25, 'y': 0.88},
              {'num': 5, 'x': 0.68, 'y': 0.84}
              ]}

    ax.setSegmentMaskNumbers(imgSegmentMask, **kwargs)

    # Get around 'plt.figtext' not being scalable with the image
    fig.savefig('numbered_segments.png')

    plt.show()


def getSegmentedImg():
    url = "https://arteagac.github.io/blog/lime_image/img/cat-and-dog.jpg"
    img = skimage.io.imread(url)

    img = skimage.transform.resize(img, (299, 299))
    img = (img - 0.5)*2  # Inception pre-processing

    imgSegmentMask = skimage.segmentation.quickshift(img, kernel_size=6,
                                                     max_dist=200, ratio=0.2)

    return imgSegmentMask

if __name__ == '__main__':
    print()
    print(f"matplotlib version: {matplotlib.__version__}")
    print()

    imgSegmentMask = getSegmentedImg()
    displayImgSegments(imgSegmentMask)
