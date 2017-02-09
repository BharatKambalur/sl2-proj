# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
from skimage import color
import numpy as np
import time

start = time.time()
image = io.imread("Dev_Img_1.png")
#image_lab = color.rgb2hsv(image)

# loop over the number of segments
numSegments = 1000
# apply SLIC and extract (approximately) the supplied number
# of segments
#print 'Superpixel Segmentation Starting'
segments = slic(image, n_segments = numSegments, sigma = 2)
#print '...Done'

# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
ax.set_axis_off()
plt.axis("off")
 
# show the plots
fig.set_size_inches(12,9)
fig.savefig('UCSD_Superpixel_Output.png',bbox_inches='tight')
plt.show()

# Save the output of the SLIC in CSV for further processing
np.save("SLIC_Dev_Img_1.npy",segments)
end = time.time()
print(end-start)