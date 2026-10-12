import os
from dotenv import load_dotenv
import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError


class S3Downloader:
    def __init__(self, client=None):
        """
        Initialize the S3 downloading class and its client. Bucket
        connection settings are read from environment variables.

        Parameters
        ----------
        client : botocore client | None
            S3 client to use instead of creating one. Intended for tests.
        """
        if client is not None:
            self.client = client
            return
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

    def list_keys(self, bucket_name, folder):
        """
        List every object key under a top-level folder of a bucket,
        following pagination (a single listing returns at most 1,000 keys).
        Prints a message if listing fails. Errors are reported by exception
        type only, so no connection details are printed.

        Parameters
        ----------
        bucket_name : str | None
            The bucket (Space) name. None or empty means it is not
            configured.

        folder : str | None
            Top-level folder name, without slashes. None or empty means it
            is not configured.

        Returns
        -------
        dict[str, str] | None
            Key relative to the folder (the `<folder>/` prefix removed) to
            ETag, for every object in the folder, or None if the folder
            could not be listed.
        """
        if not bucket_name or not folder:
            print("Could not list a folder: bucket or folder name is not configured.")
            return None
        prefix = f"{folder}/"
        keys = {}
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                for obj in page.get("Contents", []):
                    if obj["Key"].startswith(prefix):
                        keys[obj["Key"][len(prefix) :]] = obj.get("ETag", "")
        except (BotoCoreError, ClientError) as error:
            print(f"Could not list folder {folder}: {type(error).__name__}")
            return None
        return keys

    def download_file(self, bucket_name, object_name, local_filename):
        """
        Downloads a file from S3 into local storage. Prints a message
        saying whether the download succeeded. Errors are reported by
        exception type only, so no connection details are printed.

        Parameters
        ----------
        bucket_name : str | None
            The bucket (Space) name. None or empty means it is not
            configured.

        object_name : str
            Full object key, including the top-level folder

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
