import os
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import requests
model = keras.models.load_model('./models/model/model_per_class.h5')

# DEFINE seg-areas
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}

# Constants
VOLUME_SLICES = 100
VOLUME_START_AT = 22 # first slice of volume that we will include
IMG_SIZE = 128
#
def download_nifti(fileName,url):
    try:
        print(url)
        if not os.path.exists(f'./tmp/{fileName}'):
            response = requests.get(url)
            print(response)
            response.raise_for_status()  # Ensure the request was successful
            with open(f'./tmp/{fileName}', 'wb') as f:
                f.write(response.content)
        return nib.load(f'./tmp/{fileName}').get_fdata()
    except requests.exceptions.ConnectionError as err:
    # eg, no internet
        raise SystemExit(err)
    except requests.exceptions.HTTPError as err:
        # eg, url, server and other errors
        raise SystemExit(err)
    #
def predictByPath(flairObj, ceObj):
    flair = download_nifti(flairObj["public_id"], flairObj["secure_url"])
    ce = download_nifti(ceObj["public_id"], ceObj["secure_url"])
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))

    for j in range(VOLUME_SLICES):
        X[j,:,:,0] = cv2.resize(flair[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        X[j,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

    return model.predict(X/np.max(X),verbose=1)

def showPredicts(flairObj, ce, t2,result_filename, start_slice = 60):
    
    flair = download_nifti(flairObj["public_id"], flairObj["secure_url"])
    gt = download_nifti(t2["public_id"], t2["secure_url"])
    p = predictByPath(flairObj,ce)
    print(p)
    core = p[:,:,:,1]
    edema = p[:,:,:,2]
    enhancing = p[:,:,:,3]
    
    fig, axarr = plt.subplots(1, 6, figsize=(18, 5))
    
    # for i in range(6):  # for each image, add brain background
    # Create and save images
    for i in range(6):  # for each image, add brain background
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(cv2.resize(flair[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')
        
        if i == 0:
            ax.imshow(cv2.resize(flair[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
            #ax.title.set_text('Original image flair')
        elif i == 1:
            curr_gt = cv2.resize(gt[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            ax.imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3)
            #ax.title.set_text('Ground truth')
        elif i == 2:
            ax.imshow(p[start_slice, :, :, 1:4], cmap="Reds", interpolation='none', alpha=0.3)
            #ax.title.set_text('All classes')
        elif i == 3:
            ax.imshow(edema[start_slice, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
            #ax.title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
        elif i == 4:
            ax.imshow(core[start_slice, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
            #ax.title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
        elif i == 5:
            ax.imshow(enhancing[start_slice, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
            #ax.title.set_text(f'{SEGMENT_CLASSES[3]} predicted')
        # Turn off axes and frames
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        image_path =  f'./results/{result_filename}_{i}.png'
        plt.savefig(image_path,bbox_inches='tight', pad_inches=0 )
        plt.close(fig)
        plt.close()

    #     axarr[i].imshow(cv2.resize(flair[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray", interpolation='none')
    #     axarr[0].imshow(cv2.resize(flair[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    #     cv2.imwrite('original_image.png',flair[:,:,start_slice+VOLUME_START_AT] )
    #     axarr[0].title.set_text('Original image flair')
    #     curr_gt = cv2.resize(gt[:,:,start_slice+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    #     cv2.imwrite('ground_image.png',curr_gt )
    #     axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3)
    #     axarr[1].title.set_text('Ground truth')
    #     axarr[2].imshow(p[start_slice,:,:,1:4], cmap="Reds", interpolation='none', alpha=0.3)
    #     cv2.imwrite('all_image.png',p[start_slice,:,:,1:4] )
    #     axarr[2].title.set_text('All classes')
    #     axarr[3].imshow(edema[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
    #     cv2.imwrite('edema_image.png',edema[start_slice,:,:] )
    #     axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
    #     axarr[4].imshow(core[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
    #     cv2.imwrite('core_image.png',core[start_slice,:,:] )
    #     axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
    #     axarr[5].imshow(enhancing[start_slice,:,:], cmap="OrRd", interpolation='none', alpha=0.3)
    #     cv2.imwrite('enhancing_image.png',enhancing[start_slice,:,:])
    #     axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')

    plt.show()

