import logging
import azure.functions as func
from pathlib import Path    


def main(myblob: func.InputStream, msg: func.Out[str]):
    filename = Path(myblob.name).name
    logging.info(f"Processing Blob: {filename}")
    msg.set(filename)

    
