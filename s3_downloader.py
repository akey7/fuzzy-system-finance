import os
from dotenv import load_dotenv
import boto3
from botocore.client import Config


class S3Downloader:
    def __init__(self):
        """
        Initialize the S3 downloading class.
        """
        load_dotenv()
        self.region_name = os.getenv("FSF_FRONT_END_BUCKET_REGION")
        self.endpoint_url = os.getenv("FSF_FRONT_END_BUCKET_ENDPOINT")
        self.aws_access_key_id = os.getenv("FSF_FRONT_END_BUCKET_READ_ONLY_KEY_ID")
        self.aws_secret_access_key = os.getenv("FSF_FRONT_END_BUCKET_READ_ONLY")

    def download_file(self, bucket_name, object_name, local_filename):
        """
        Downloads a file from S3 into local storage. Prints a message when
        it downloads.

        Parameters
        ----------
        bucket_name : str
            The bucket name.

        object_name : str
            Object name

        local_filename : str
            Absolute path to the local filename
        """
        session = boto3.session.Session()
        client = session.client(
            "s3",
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            config=Config(signature_version="s3v4"),
        )
        client.download_file(bucket_name, object_name, local_filename)
        print(f"Downloaded {local_filename}!")
