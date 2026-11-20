"""Training commands fail with the documented message without the extra."""

from __future__ import annotations

import pytest

from stc import bootstrap
from stc.domain.errors import MissingExtraError


def test_require_train_extra_raises_the_documented_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "_module_available", lambda name: False)

    with pytest.raises(MissingExtraError) as info:
        bootstrap.require_train_extra()

    assert info.value.exit_code == 2
    assert "training needs the 'train' extra" in info.value.message
    assert "spanish-text-classifier-lora[train]" in info.value.message


def test_build_trainer_checks_the_extra_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "_module_available", lambda name: False)

    with pytest.raises(MissingExtraError):
        bootstrap.build_trainer("lora")


def test_repository_identity_returns_hashes() -> None:
    git_sha, lock_sha = bootstrap.repository_identity()

    assert git_sha
    assert len(lock_sha) == 64
