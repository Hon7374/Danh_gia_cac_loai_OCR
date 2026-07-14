from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from typing import Iterator

from app.config import OCR_TEMP_DIR


@contextmanager
def workspace_temporary_directory(prefix: str = "ocr_tmp_") -> Iterator[str]:
    """Create an inherited-ACL temp directory inside the project workspace.

    ``tempfile.TemporaryDirectory`` applies a restrictive Windows ACL that can
    become non-traversable when the demo is launched through a managed user or
    service identity.  A normal directory inherits the verified workspace ACL
    and is still cleaned up on every exit path.
    """
    OCR_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    safe_prefix = "".join(char for char in str(prefix) if char.isalnum() or char in {"-", "_"})
    path = OCR_TEMP_DIR / f"{safe_prefix or 'ocr_tmp_'}{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)
