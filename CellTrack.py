# Code for detecting & tracking cells

from __future__ import division, unicode_literals, print_function

import os.path
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("figure", figsize=(10, 5))
matplotlib.rc('image', cmap='gray')
import numpy as np
import pims
import trackpy as tp


@pims.pipeline()
def gray(image):
    return image[:, :, 1]


# Change video to gray scale
frames = gray(pims.open("file:C:/Users/kingf/Desktop/videoalgea/221113sleepy.avi"))
# f=tp.locate(frames[0],11,invert=True)
# with open("frames",'w') as f:
#    for x in range(1,2):
#        f.write(f"{frames[x]}\n")
# if os.path.exists("frames"):
#    print("Frames file create!!")
# else:
#    print("file not found!")

# plt.imshow(frames[0]);
# frames[123].frame_no

# Create data by extracting into frames
with open("table", 'w') as f:
    for y in range(1, 5):
        table1 = tp.locate(frames[y], 11, invert=True)
        f.write(f"{table1}\n")
if os.path.exists("table"):
    print('table create!')
else:
    print('go kill yourself!')

# create a graph that locate the particle
tp.annotate(table1, frames[4]);

# refine parameters to eliminate spurious features (bug)
fig, ax = plt.subplot()
ax.hist(table1['mass'], bins=20)

# after refinement
table1 = tp.locate(frames[0], 11, invert=True, minmass=20)
tp.annotate(table1, frames[0]);

# check for subpixel accuracy
tp.subpx_bias(table1)
tp.subpx_bias(tp.locate(frames[0], 7, invert=True, minmass=20));

# collect all data from set frames to a table
table2 = tp.batch(frames[:5], 11, minmass=20, invert=True);

# link features into trajectories
table3 = tp.link(table1, 5, memory=3)
np.set_printoptions(threshold=np.inf)
with open("link", 'w') as f:
    for y in range(1, 5):
        table1 = tp.locate(frames[y], 11, invert=True)
        f.write(f"{table3}\n")
if os.path.exists("link"):
    print('link create!')
else:
    print('go kill yourself!')

t1 = tp.filter_stubs(table3, 25)
print('Before:', table3['particle'].nunique())
print('After:', t1['particle'].nunique())
plt.figure()
tp.mass_size(t1.groupby('particle').mean());
t2 = t1[((t1['mass'] > 50) & (t1['size'] < 2.6) & (t1['ecc'] < 0.3))]
plt.figure()
tp.annotate(t2[t2['frame'] == 0], frames[0]);
plt.figure()
tp.plot_traj(t2);
