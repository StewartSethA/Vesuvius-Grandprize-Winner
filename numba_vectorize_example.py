import numpy as np
from numba import vectorize

@vectorize
def generate_xyxys_ids(fragment_id, image, mask, fragment_mask, tile_size, size, stride, is_valid=False):
        xyxys = []
        ids = []
        x1_list = list(range(0, image.shape[1]-tile_size+1, stride))
        y1_list = list(range(0, image.shape[0]-tile_size+1, stride))
        #windows_dict={}
        for a in y1_list:
            for b in x1_list:
                for yi in range(0,tile_size,size):
                    for xi in range(0,tile_size,size):
                        y1=a+yi
                        x1=b+xi
                        y2=y1+size
                        x2=x1+size
                # for y2 in range(y1,y1 + tile_size,size):
                #     for x2 in range(x1, x1 + tile_size,size):
                        if not is_valid:
                            if not np.all(np.less(mask[a:a + tile_size, b:b + tile_size],0.01)):
                                if not np.any(np.equal(fragment_mask[a:a+ tile_size, b:b + tile_size],0)):
                                    # if (y1,y2,x1,x2) not in windows_dict:
                                    #train_images.append(image[y1:y2, x1:x2])
                                    xyxys.append([x1,y1,x2,y2])
                                    ids.append(fragment_id)
        return xyxys, ids

images = np.ones((8192, 8192, 100), dtype=np.uint8)
mask = np.ones((8192, 8129), dtype=np.uint8)
fragment_mask = np.ones((8192, 8129), dtype=np.uint8)
