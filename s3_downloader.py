import os
from dotenv import load_dotenv
import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError


class S3Downloader:
    def __init__(self):
        """
        Initialize the S3 downloading class and its client. Bucket
        connection settings are read from environment variables.
        """
        load_dotenv()
        session = boto3.session.Session()
        self.client = session.client(
            "s3",
            region_name=os.getenv("FSF_FRONT_END_BUCKET_REGION"),
            endpoint_url=os.getenv("FSF_FRONT_END_BUCKET_ENDPOINT"),
            aws_access_key_id=os.getenv("FSF_FRONT_END_BUCKET_READ_ONLY_KEY_ID"),
            aws_secret_access_key=os.getenv("FSF_FRONT_END_BUCKET_READ_ONLY"),
            config=Config(signature_version="s3v4"),
        )

    def download_file(self, bucket_name, object_name, local_filename):
        """
        Downloads a file from S3 into local storage. Prints a message
        saying whether the download succeeded. Errors are reported by
        exception type only, so no connection details are printed.

        Parameters
        ----------
        bucket_name : str | None
            The bucket name. None or empty means it is not configured.

        object_name : str
            Object name

        local_filename : str
            Path to the local filename

        Returns
        -------
        bool
            True if the file was downloaded, False otherwise.
        """
        if not bucket_name:
            print(f"Could not download {object_name}: bucket name is not configured.")
            return False
        try:
            self.client.download_file(bucket_name, object_name, local_filename)
        except (BotoCoreError, ClientError, OSError) as error:
            print(f"Could not download {object_name}: {type(error).__name__}")
            return False
        print(f"Downloaded {local_filename}!")
        return True
