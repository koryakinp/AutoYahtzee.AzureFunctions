from azure.storage.blob import BlobServiceClient


class StorageAccess:

    def __init__(self, cs):

        blob_service_client = BlobServiceClient.from_connection_string(cs)

        self.prediction_container = blob_service_client.get_container_client(
            "autoyahtzee-predictions")
        self.webm_container = blob_service_client.get_container_client(
            "autoyahtzee-processed-video-container")
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


