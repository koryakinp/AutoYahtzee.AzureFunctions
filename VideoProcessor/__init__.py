import logging
import os
import glob
import numpy as np
import tempfile
import pathlib

from .image_processing import predict
from .image_processing import process_image
from .image_processing import get_dice_images_for_human
from .video_processing import process_video
from ..shared.data_access import DataAccess
from ..shared.storage_access import StorageAccess
from ..shared.utils import get_throw_id
from tensorflow import keras

import azure.functions as func

logging.getLogger("azure.storage.common.storageclient").setLevel(logging.WARNING)

da = None
sa = None
model = None

def main(
    msg: func.QueueMessage,
    inputblob: func.InputStream) -> None:

    payload = msg.get_body().decode('utf-8')
    throw_id = get_throw_id(payload)
    logging.info('Queue Message Payload: %s', payload)

    global da
    if da is None:
        da = DataAccess(os.environ['AZURE_SQL_CONNECTION_STRING'])

    global sa
    if sa is None:
        sa = StorageAccess(os.environ['AZURE_STORAGE_CONNECTION_STRING'])

    global model
    if model is None:
        parent = pathlib.Path(__file__).parent
        path = os.path.join(parent, 'model', 'autoyahtzee.h5')
        model = keras.models.load_model(path)

    prediction_ids = da.get_predictions(throw_id)
    throw_ids =  da.get_throw(throw_id)

    if len(prediction_ids) > 0 or len(throw_ids) > 0:
        logging.info('Blob was processed earlier')

        logging.info('Removing records from a blob storage')
        sa.clear_blobs(throw_id, prediction_ids)

        logging.info('Removing records from a SQL storage')
        da.delete_throw(throw_id)

    tf = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

    with open(tf.name, "wb") as f:
        f.write(inputblob.read())    

    logging.info('Savinsg Throw ' + throw_id + ' to SQL')
    da.insert_throw(throw_id)

    logging.info('Processing Video %s', tf.name)
    mp4, webm, jpg = process_video(tf.name)

    jpgs = glob.glob("/home/site/wwwroot/*.jpg")
    mp4s = glob.glob("/home/site/wwwroot/*.mp4")
    webms = glob.glob("/home/site/wwwroot/*.webm")

    logging.info('JPGs: ' + str(len(jpgs)))
    logging.info('MP4s: ' + str(len(mp4s)))
    logging.info('WEBMs: ' + str(len(webms)))

    logging.info('Uploading ' + throw_id + '.mp4')
    sa.upload_mp4(os.path.abspath(mp4), throw_id)

    logging.info('Uploading ' + throw_id + '.webm')
    sa.upload_webm(os.path.abspath(webm), throw_id)

    logging.info('Uploading ' + throw_id + '.jpg')
    sa.upload_image(os.path.abspath(jpg), throw_id)

    logging.info('Processing Image')
    dice_images, peaks = process_image(jpg)

    logging.info('Running Prediction')
    prediction, confidence, label = predict(dice_images, model)

    logging.info('Extracting Dice images')
    di = get_dice_images_for_human(jpg, peaks)

    logging.info('Saving to Azure Container')
    for image_to_upload in di:
        sa.upload_prediction(os.path.abspath(image_to_upload.name))

    logging.info('Saving to SQL')
    da.insert_prediction(throw_id, prediction, confidence)
