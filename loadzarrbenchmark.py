import zarr

import time

zarr_path = "train_scrolls/20231210132040/20231210132040_64.zarr"
zarr_path = "train_scrolls/20231210132040/20231210132040_128.zarr"
t = time.time()
mra = zarr.open(zarr_path, mode="r")
print(time.time()-t, mra.info)
t = time.time()
ra0 = mra[0]
print(time.time()-t, ra0.info)
#t = time.time()
#slice = ra0[25]
#print("time to load full z slice", time.time()-t, slice.shape)
t = time.time()
#block = ra0.blocks[10,10,10]
block = ra0.blocks[0,0,0]
print("time to load 3d block", time.time()-t, block.shape)

#def slice(x1,x2,y1,y2)

t = time.time()
#print(ra0[:,65:256,128:256].shape)
import random
# (65, 8790, 12122)
x1 = random.randint(0,8790-64)
y1 = random.randint(0,12122-64)

print(ra0.shape)
print(ra0.__class__)
#print(ra0[0:65,x1:x1+64,y1:y1+64].shape)
print(time.time()-t, "elapsed")

chunkshape = [d for d in ra0.info_items() if d[0] == "Chunk shape"][0][1]
shape = [d for d in ra0.info_items() if d[0] == "Shape"][0][1]
