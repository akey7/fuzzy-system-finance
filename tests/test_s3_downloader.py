"""
Tests for S3Downloader.list_keys() with a fake S3 client. No network access
and no credentials are used.
"""

from botocore.exceptions import ClientError

from s3_downloader import S3Downloader


class FakePaginator:
    """
    Stand-in for a boto3 list_objects_v2 paginator that yields fixed pages.
    """

    def __init__(self, pages, error=None):
        self.pages = pages
        self.error = error
        self.calls = []

    def paginate(self, **kwargs):
        self.calls.append(kwargs)
        for page in self.pages:
            yield page
        if self.error is not None:
            raise self.error


class FakeClient:
    """
    Stand-in for a boto3 S3 client that only supports get_paginator().
    """

    def __init__(self, paginator):
        self.paginator = paginator

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2"
        return self.paginator


def test_list_keys_follows_all_pages_and_strips_folder():
    pages = [
        {
            "Contents": [
                {"Key": "time-series/2026-10-08/all_forecasts.csv", "ETag": '"a"'},
                {"Key": "time-series/all_forecasts.csv", "ETag": '"b"'},
            ]
        },
        {"Contents": [{"Key": "time-series/2026-10-09/all_forecasts.csv", "ETag": '"c"'}]},
    ]
    paginator = FakePaginator(pages)
    downloader = S3Downloader(client=FakeClient(paginator))
    keys = downloader.list_keys("space", "time-series")
    assert keys == {
        "2026-10-08/all_forecasts.csv": '"a"',
        "all_forecasts.csv": '"b"',
        "2026-10-09/all_forecasts.csv": '"c"',
    }
    assert paginator.calls == [{"Bucket": "space", "Prefix": "time-series/"}]


def test_list_keys_ignores_keys_outside_folder():
    # S3 applies the prefix itself; this guards against keys such as
    # "time-series-old/..." or keys from a client that ignores Prefix.
    pages = [
        {
            "Contents": [
                {"Key": "time-series-old/2026-10-09/all_forecasts.csv", "ETag": '"a"'},
                {"Key": "portfolio-optimization/2026-10-09/optimization_metadata.yml"},
                {"Key": "time-series/2026-10-09/all_forecasts.csv", "ETag": '"c"'},
            ]
        }
    ]
    downloader = S3Downloader(client=FakeClient(FakePaginator(pages)))
    assert downloader.list_keys("space", "time-series") == {
        "2026-10-09/all_forecasts.csv": '"c"'
    }


def test_list_keys_empty_folder():
    downloader = S3Downloader(client=FakeClient(FakePaginator([{"KeyCount": 0}])))
    assert downloader.list_keys("space", "time-series") == {}


def test_list_keys_error_returns_none():
    error = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "ListObjectsV2"
    )
    downloader = S3Downloader(client=FakeClient(FakePaginator([], error=error)))
    assert downloader.list_keys("space", "time-series") is None


def test_list_keys_unconfigured_bucket_or_folder():
    downloader = S3Downloader(client=FakeClient(FakePaginator([])))
    assert downloader.list_keys(None, "time-series") is None
    assert downloader.list_keys("space", None) is None
    assert downloader.list_keys("space", "") is None
