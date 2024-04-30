import tarfile
import os
import glob
import atomsci.ddm.utils.file_utils as futils

from tempfile import TemporaryDirectory
import pytest
import atomsci.ddm.utils.test_utils as tu


@pytest.mark.basic
def test_extractall_tarfile():
    tar_file = tu.relative_to_file(
        __file__, "../../examples/BSEP/models/bsep_classif_scaffold_split.tar.gz"
    )
    with TemporaryDirectory() as extract_location:
        with tarfile.open(tar_file, mode="r:gz") as tar:
            futils.safe_extract(tar, path=extract_location)
            print("extract_location = " + extract_location)
            print(glob.glob(extract_location + "/*"))
            assert len(os.listdir(extract_location)) > 0
