import logging
import azure.functions as func


def main(myblob: func.InputStream, msg: func.Out[str]):
    logging.info(f"Processing Blob: {myblob.name}")
    msg.set(myblob.name)

    
