"""빈 평가 구간은 형식 검사 성공으로 처리하지 않는다."""
from importlib import import_module
from src.evaluate import check


def test_empty_days_fail_without_predicting(capsys):
    class Model:
        def predict(self, day):
            raise AssertionError("빈 구간에서 예측하면 안 됨")

    assert check(Model(), []) == 1
    output = capsys.readouterr().out
    assert "검사할 날이 없음" in output
    assert "형식 확인 완료" not in output


def test_empty_dataset_split_fails(monkeypatch, capsys):
    class EmptyDataset:
        def __init__(self, *args, **kwargs):
            pass

        def split(self, boundary):
            assert boundary == "2026-06-01"
            return [], []

    monkeypatch.setattr(import_module("src.evaluate"), "Dataset", EmptyDataset)
    assert check(object(), "2026-06-01", data="dataset") == 1
    assert "--split" in capsys.readouterr().out
