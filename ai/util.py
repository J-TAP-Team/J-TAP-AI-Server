import boto3
from .config import s3_config

def s3upload(file, filename):

    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=s3_config["AccessKeyId"],
            aws_secret_access_key=s3_config["SecretKey"],
        )

        region = s3_config["region"]

        bucket_name = s3_config["bucket_name"]

        s3.upload_fileobj(
            file, bucket_name, filename, ExtraArgs={"ACL": "public-read"}
        )

        url = "https://s3-%s.amazonaws.com/%s/%s" % (region, bucket_name, filename)

    except Exception as e:
        return e

    return url