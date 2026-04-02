"""Paths and target definitions (documented in README)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "Subsidy Report 2025.xlsx"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Статусы заявки → целевая переменная outcome_positive (см. README)
POSITIVE_STATUSES = frozenset(
    {
        "Исполнена",
        "Одобрена",
        "Сформировано поручение",
        "Получена",
    }
)
NEGATIVE_STATUSES = frozenset({"Отклонена"})
# Отозвано: исключаем из обучения бинарной классификации (заявитель сам отозвал)
EXCLUDED_STATUSES = frozenset({"Отозвано"})

RANDOM_STATE = 42
