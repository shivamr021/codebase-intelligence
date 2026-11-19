"""tests/test_walker.py — qf-chunk-id: React test files skipped (seam C: is_test_file, walk_repo on tmp_path)."""
import pytest
from core.ingestion.walker import is_test_file, walk_repo


@pytest.mark.parametrize("filename", ["Button.test.tsx", "Card.spec.tsx", "form.test.jsx", "Modal.spec.jsx",
                                      "Button.Test.TSX", "Nav.SPEC.Jsx"])
def test_react_test_files_count_as_test_files(filename):
    # Middle, new. Break: walker.py:107-114 lists only .test/.spec + .js/.ts, so React tests get indexed as product code.
    assert is_test_file(filename) is True


@pytest.mark.parametrize("filename", ["Button.tsx", "contest.tsx", "latest.jsx", "spec.jsx"])
def test_plain_react_files_stay_non_test(filename):
    # Edge, existing guard. Break: an over-wide fix (endswith ".tsx"/"test.tsx") drops real components from the index.
    assert is_test_file(filename) is False


def test_walker_lists_app_but_not_its_react_test(tmp_path):
    # Edge, new. Break: walker.py:171 lets App.test.tsx through, so the index holds the test beside the component.
    (tmp_path / "src").mkdir()
    for name in ("App.tsx", "App.test.tsx"):
        (tmp_path / "src" / name).write_text("export const App = () => null;\n")
    assert [f["rel_path"] for f in walk_repo(str(tmp_path))] == ["src/App.tsx"]
