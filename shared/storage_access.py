from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError


class StorageAccess:

    def __init__(self, cs):

        blob_service_client = BlobServiceClient.from_connection_string(cs)

        self.prediction_container = blob_service_client.get_container_client(
            "autoyahtzee-predictions")
        self.webm_container = blob_service_client.get_container_client(
            "autoyahtzee-processed-video-container-webm")
        self.mp4_container = blob_service_client.get_container_client(
            "autoyahtzee-processed-video-container-mp4")
        self.jpg_container = blob_service_client.get_container_client(
            "autoyahtzee-processed-image-container")

    def upload_prediction(self, filename):
        with open(filename, "rb") as data:
            self.prediction_container.upload_blob(filename, data)

    def upload_mp4(self, filename, throw_id):
        with open(filename, "rb") as data:
            self.mp4_container.upload_blob(throw_id + '.mp4', data)

    def upload_webm(self, filename, throw_id):
        with open(filename, "rb") as data:
            self.webm_container.upload_blob(throw_id + '.webm', data)

    def upload_image(self, filename, throw_id):
        with open(filename, "rb") as data:
            self.jpg_container.upload_blob(throw_id + '.jpg', data)

    def clear_blobs(self, throw_id, prediction_ids):
        self.delete_mp4_if_exists(throw_id)
        self.delete_webm_if_exists(throw_id)
        self.delete_jpg_if_exists(throw_id)

        for prediction_id in prediction_ids:
            self.delete_prediction_if_exists(prediction_id)

    def delete_mp4_if_exists(self, throw_id):
        blob_name = throw_id + '.mp4'
        if self.check_if_blob_exists(blob_name, self.mp4_container):
            self.mp4_container.delete_blob(throw_id + '.mp4')

    def delete_webm_if_exists(self, throw_id):
        blob_name = throw_id + '.webm'
        if self.check_if_blob_exists(blob_name, self.webm_container):
            self.webm_container.delete_blob(blob_name)

    def delete_jpg_if_exists(self, throw_id):
        blob_name = throw_id + '.jpg'
        if self.check_if_blob_exists(blob_name, self.jpg_container):
            self.jpg_container.delete_blob(blob_name)

    def delete_prediction_if_exists(self, prediction_id):
        blob_name = prediction_id + '.jpg'
        if self.check_if_blob_exists(blob_name, self.prediction_container):
            self.prediction_container.delete_blob(blob_name)


    def check_if_blob_exists(self, blob_name, container):
        try:
            blob_properties = container.get_blob_client(blob_name).get_blob_properties()
            return True
        except ResourceNotFoundError as ex:
            return False


