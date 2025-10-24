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
#video_path = '/mnt/c/Users/kingf/Desktop/videoalgea/test9-230918 x6.avi'
video_path = 'C:/Users/kingf/Desktop/videoalgea/221113sleepy.avi'
# Load the video and convert it to grayscale
if os.path.exists(video_path):
    frames = gray(pims.open(video_path))
    print('Video loaded successfully!')
else:
    print('Video file not found:', video_path)
# Create a file to store the detected cell data

# show the scan frame
f = tp.locate(frames[0], 11, invert=True)
tp.annotate(f, frames[0]);

# show the mass which includes noise
#fig, ax = plt.subplot()
#ax.hist(table1['mass'], bins=20)

# Optionally, label the axes.
#ax.set(xlabel='mass', ylabel='count');

# Refinement
table1 = tp.locate(frames[0], 11, invert=True, minmass=1000)
tp.annotate(table1, frames[0])

# Check for subpixel accuracy
tp.subpx_bias(table1)
tp.subpx_bias(tp.locate(frames[0], 9, invert=True, minmass=1000))

# Collect all data from the set of frames into a table
table2 = tp.batch(frames[:300], 11, minmass=1000, invert=True,processes=1)

# Link features into trajectories
table3 = tp.link(table2, 5, memory=3)

# Filter out short trajectories
t1 = tp.filter_stubs(table3, 25)

print('Before:', table3['particle'].nunique())
print('After:', t1['particle'].nunique())

# Plot mass vs. size of particles
plt.figure()
tp.mass_size(t1.groupby('particle').mean())

# Filter particles based on mass, size, and eccentricity
t2 = t1[((t1['mass'] > 1000) & (t1['size'] < 10.6) & (t1['ecc'] < 0.3))]

# Annotate and plot trajectories
plt.figure()
tp.annotate(t2[t2['frame'] == 0], frames[0])
plt.figure()
tp.plot_traj(table3)

# Save the final tracking data
table3.to_csv('cell_tracking_results.csv', index=False)
