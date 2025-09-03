import os
import datajoint as dj
from pathlib import Path


logger = dj.logger
s3_session, s3_bucket = None, None


def _get_axon_s3_session():
    import djsciops.authentication as dj_auth
    import djsciops.settings as dj_settings

    global s3_session, s3_bucket
    if s3_session is not None:
        return s3_session, s3_bucket

    dj_sciops_config = dj_settings.get_config()
    s3_session = dj_auth.Session(
        aws_account_id=dj_sciops_config["aws"]["account_id"],
        s3_role=dj_sciops_config["s3"]["role"],
        auth_client_id=dj_sciops_config["djauth"]["client_id"],
        auth_client_secret=dj_sciops_config["djauth"].get("client_secret"),
    )
    s3_bucket = dj_sciops_config["s3"]["bucket"]
    return s3_session, s3_bucket


ROOT_DIR = Path("/ceph/aeon")
DB_PREFIX = "ucl-swc_aeon_"


def _upload_session_data(relative_dir):
    """
    Routine to upload data from a local directory to the Axon S3 bucket.
    """
    import djsciops.axon as dj_axon

    s3_session, s3_bucket = _get_axon_s3_session()
    relative_dir = Path(relative_dir).as_posix()

    local_session_dir = ROOT_DIR / relative_dir
    assert local_session_dir.exists(), f"{local_session_dir} does not exist"
    assert local_session_dir.is_dir(), f"{local_session_dir} is not a directory"

    dj_session_dir = f"{DB_PREFIX[:-1]}/inbox/{relative_dir}/"
    dj_axon.upload_files(
        source=local_session_dir.as_posix(),
        destination=dj_session_dir,
        session=s3_session,
        s3_bucket=s3_bucket,
    )

    remote_files = [
        (
            Path(x["key"]).relative_to(f"{DB_PREFIX[:-1]}/inbox").as_posix(),
            x["_size"],
        )
        for x in dj_axon.list_files(
            session=s3_session,
            s3_bucket=s3_bucket,
            s3_prefix=dj_session_dir,
            as_tree=False,
        )
    ]

    local_files = []
    for f in local_session_dir.rglob("[!.]*"):
        if f.is_file():
            local_files.append(
                (
                    f.relative_to(ROOT_DIR).as_posix(),
                    f.stat().st_size,
                )
            )

    if set(local_files) != set(remote_files):
        logger.warning(f"Incomplete data upload for {relative_dir} - try again")


def upload_raw():
    raw_data_dir = Path("/ceph/aeon/aeon/data/raw/AEON4/social0.2")
    subfolders = sorted([
        f.relative_to(ROOT_DIR) for f in raw_data_dir.iterdir() if f.is_dir() and not f.name.startswith(".")
    ])

    relative_dir = subfolders[4]

    _upload_session_data(relative_dir)


def upload_ingest():
    raw_data_dir = Path("/ceph/aeon/aeon/data/ingest/AEON3/social0.2")
    subfolders = sorted([
        f.relative_to(ROOT_DIR) for f in raw_data_dir.iterdir() if f.is_dir() and not f.name.startswith(".")
    ])

    relative_dir = subfolders[-1]

    _upload_session_data(relative_dir)

    # SLEAP models
    dir_222 = "aeon/data/ingest/222"
    _upload_session_data(dir_222)
