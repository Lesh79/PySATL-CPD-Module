"""
Терминология
-----------
- segment (сегмент): участок временного ряда между двумя точками разладки.
- bisegment (бисегмент): два последовательных сегмента, между которыми есть точка разладки.

Dataset
-------
Dataset — набор аннотированных временных рядов.

Состав:
1) list[PandasLabeledDataProvider] — по одному провайдеру на временной ряд.
2) timeseries_preprocessor: Callable[[pd.DataFrame], pd.DataFrame] — препроцессор,
   применяемый к каждому временному ряду при загрузке.

Основные операции:
1) load_from_dir(self, dir_path: Path) -> Dataset
   Загружает датасет из директории заданной структуры.
   При загрузке каждого ряда применяется timeseries_preprocessor.

2) filter_by_annotation(annotation_filter: Callable[[Annotation], bool]) -> Dataset
   Возвращает новый Dataset, отфильтрованный по аннотации.

3) select_bisegments_by_filter(
       filter: Optional[Callable[[tuple[SegmentInfo, SegmentInfo]], bool]]
   ) -> list[PandasLabeledDataProvider]
   Возвращает бисегменты, удовлетворяющие фильтру (для режима NoReset).
   Если filter is None, выбираются все бисегменты.

4) property timeserieses -> list[PandasLabeledDataProvider]
   Список временных рядов для режима Reset.

PandasLabeledDataProvider
-------------------------
PandasLabeledDataProvider — наследник LabeledDataProvider[NumericArray].

Принимает:
1) pd.DataFrame временного ряда с колонкой segments (идентификатор сегмента).
2) DatasetSegmentInfo — таблицу описаний сегментов (минимум: начало и конец сегмента;
   также могут быть произвольные пользовательские поля).
3) Annotation — dataclass с аннотацией временного ряда
   (например: путь, сценарий режима, версия).

Основные операции:
1) change_point
   Автоматически определяется по колонке segments.

2) __iter__() -> Iterator[NumericArray]
   Итерирует по строкам self.dataset без колонки segments.

3) select_columns(columns: list[str]) -> PandasLabeledDataProvider
   Возвращает новый провайдер с выбранными колонками в порядке columns.

4) query_bisegments_indexes(
       filter: Optional[Callable[[tuple[SegmentInfo, SegmentInfo]], bool]]
   ) -> list[tuple[int, int, int]]
   Возвращает индексы в формате:
   (начало текущего сегмента, точка разладки, конец следующего сегмента).

5) query_bisegments(
       filter: Optional[Callable[[tuple[SegmentInfo, SegmentInfo]], bool]]
   ) -> list[PandasLabeledDataProvider]
   Возвращает список провайдеров, каждый из которых соответствует одному бисегменту.

Важно
-----
При формировании нового PandasLabeledDataProvider необходимо сбрасывать индекс DataFrame
(reset_index(drop=True)). В проекте используется только дефолтный индекс (номер строки).
Если берется подтаблица, индекс должен быть приведен к непрерывному.
"""

__author__ = "Andrey Isakov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.core.typedefs import NumericArray

SEGMENT_COLUMN = "segment"
SEGMENT_ID_COLUMN = "segment"
SEGMENT_START_COLUMN = "start"
SEGMENT_END_COLUMN = "end"

type DatasetSegmentInfo = pd.DataFrame
type SegmentFilter = Callable[[tuple[SegmentInfo, SegmentInfo]], bool]
type AnnotationFilter = Callable[[Annotation], bool]
type TimeseriesPreprocessor = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True, kw_only=True)
class Annotation:
    """
    Metadata descriptor for a single annotated time series.
    """

    path: str | None = None
    scenario: str | None = None
    version: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class SegmentInfo:
    """
    Segment metadata row used for bisegment filtering.
    """

    segment: int | str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("Segment start index must be non-negative")
        if self.end < self.start:
            raise ValueError("Segment end index must be greater than or equal to segment start index")


class PandasLabeledDataProvider(LabeledData[NumericArray]):
    """
    DataProvider implementation for a single annotated time series.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        segment_info: DatasetSegmentInfo,
        annotation: Annotation,
        name: str | None = None,
    ) -> None:
        if SEGMENT_COLUMN not in dataset.columns:
            raise ValueError(f"Dataset must contain '{SEGMENT_COLUMN}' column")

        required_segment_columns = {SEGMENT_START_COLUMN, SEGMENT_END_COLUMN}
        if not required_segment_columns.issubset(segment_info.columns):
            missing_columns = required_segment_columns.difference(segment_info.columns)
            raise ValueError(f"Segment info is missing required columns: {sorted(missing_columns)}")

        self._dataset = dataset.copy().reset_index(drop=True)
        self._segment_info = segment_info.copy().reset_index(drop=True)
        self._annotation = annotation

        self._segment_info = self._normalize_segment_info()
        self._validate_segment_ranges()

        raw_data = self._dataset.loc[:, self.feature_columns].to_numpy(dtype=np.float64, copy=False)
        super().__init__(raw_data=raw_data, change_points=self.change_point, name=name)

    def __iter__(self) -> Iterator[NumericArray]:
        feature_values = self._dataset.loc[:, self.feature_columns].to_numpy(dtype=np.float64, copy=False)
        return iter(feature_values)

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def dataset(self) -> pd.DataFrame:
        return self._dataset.copy()

    @property
    def segment_info(self) -> DatasetSegmentInfo:
        return self._segment_info.copy()

    @property
    def annotation(self) -> Annotation:
        return self._annotation

    @property
    def feature_columns(self) -> list[str]:
        return [column for column in self._dataset.columns if column != SEGMENT_COLUMN]

    @property
    def change_point(self) -> tuple[int, ...]:
        segments = self._dataset[SEGMENT_COLUMN].to_numpy(copy=False)
        if len(segments) <= 1:
            return ()

        change_points = np.flatnonzero(segments[1:] != segments[:-1]) + 1
        return tuple(int(position) for position in change_points.tolist())

    def select_columns(self, columns: Sequence[str]) -> "PandasLabeledDataProvider":
        requested_columns = [column for column in columns if column != SEGMENT_COLUMN]
        unknown_columns = set(requested_columns).difference(self.feature_columns)
        if unknown_columns:
            raise ValueError(f"Unknown feature columns requested: {sorted(unknown_columns)}")
        if not requested_columns:
            raise ValueError("At least one feature column must be selected")

        selected_dataset = self._dataset.loc[:, [*requested_columns, SEGMENT_COLUMN]].copy().reset_index(drop=True)
        selected_segment_info = self._segment_info.copy().reset_index(drop=True)

        return PandasLabeledDataProvider(
            dataset=selected_dataset,
            segment_info=selected_segment_info,
            annotation=self._annotation,
            name=self.name,
        )

    # TODO: BiSegmentFilter naming
    def query_bisegments_indexes(self, filter_fn: SegmentFilter | None = None) -> list[tuple[int, int, int]]:
        return [
            (current.start, next_segment.start, next_segment.end)
            for current, next_segment in self._iter_segment_pairs(filter_fn)
        ]

    def query_bisegments(self, filter_fn: SegmentFilter | None = None) -> list["PandasLabeledDataProvider"]:
        result: list[PandasLabeledDataProvider] = []
        for current, next_segment in self._iter_segment_pairs(filter_fn):
            sliced_dataset = self._dataset.iloc[current.start : next_segment.end + 1].copy().reset_index(drop=True)
            split_index = next_segment.start - current.start

            sliced_segment_info = pd.DataFrame(
                [
                    {
                        SEGMENT_ID_COLUMN: current.segment,
                        SEGMENT_START_COLUMN: 0,
                        SEGMENT_END_COLUMN: split_index - 1,
                        **current.attributes,
                    },
                    {
                        SEGMENT_ID_COLUMN: next_segment.segment,
                        SEGMENT_START_COLUMN: split_index,
                        SEGMENT_END_COLUMN: next_segment.end - current.start,
                        **next_segment.attributes,
                    },
                ]
            ).reset_index(drop=True)

            result.append(
                PandasLabeledDataProvider(
                    dataset=sliced_dataset,
                    segment_info=sliced_segment_info,
                    annotation=self._annotation,
                    name=f"{self.name}:{current.segment}->{next_segment.segment}",
                )
            )

        return result

    def _normalize_segment_info(self) -> DatasetSegmentInfo:
        unique_segments = self._dataset[SEGMENT_COLUMN].drop_duplicates().tolist()
        normalized_info = self._segment_info.copy()

        if SEGMENT_ID_COLUMN in normalized_info.columns:
            normalized_rows: list[pd.Series[Any]] = []
            for segment_id in unique_segments:
                rows_for_segment = normalized_info[normalized_info[SEGMENT_ID_COLUMN] == segment_id]
                if rows_for_segment.empty:
                    raise ValueError(f"Missing segment info row for segment '{segment_id}'")
                normalized_rows.append(rows_for_segment.iloc[0])
            return pd.DataFrame(normalized_rows).reset_index(drop=True)

        if len(normalized_info) < len(unique_segments):
            raise ValueError(
                "Segment info must contain at least as many rows as unique segments in the dataset "
                f"({len(unique_segments)})"
            )

        normalized_info = normalized_info.iloc[: len(unique_segments)].copy().reset_index(drop=True)
        normalized_info.insert(0, SEGMENT_ID_COLUMN, unique_segments)
        return normalized_info

    def _validate_segment_ranges(self) -> None:
        data_length = len(self._dataset)
        if data_length == 0:
            return

        for _, segment_row in self._segment_info.iterrows():
            start = int(segment_row[SEGMENT_START_COLUMN])
            end = int(segment_row[SEGMENT_END_COLUMN])
            if start < 0:
                raise ValueError(f"Segment start index cannot be negative. Found: {start}")
            if end < start:
                raise ValueError(f"Segment end index must be >= start index. Found: start={start}, end={end}")
            if end >= data_length:
                raise ValueError(f"Segment end index {end} exceeds dataset length {data_length}")

    def _iter_segment_pairs(self, filter_fn: SegmentFilter | None) -> list[tuple[SegmentInfo, SegmentInfo]]:
        segments = self._segment_infos()
        segment_pairs: list[tuple[SegmentInfo, SegmentInfo]] = []

        for idx in range(len(segments) - 1):
            pair = (segments[idx], segments[idx + 1])
            if filter_fn is None or filter_fn(pair):
                segment_pairs.append(pair)

        return segment_pairs

    def _segment_infos(self) -> list[SegmentInfo]:
        segment_infos: list[SegmentInfo] = []
        for _, row in self._segment_info.iterrows():
            row_dict = row.to_dict()
            start = int(row_dict.pop(SEGMENT_START_COLUMN))
            end = int(row_dict.pop(SEGMENT_END_COLUMN))
            segment = row_dict.pop(SEGMENT_ID_COLUMN)

            segment_infos.append(
                SegmentInfo(
                    segment=int(segment) if isinstance(segment, np.integer) else segment,
                    start=start,
                    end=end,
                    attributes=row_dict,
                )
            )
        return segment_infos


class Dataset(Sequence[PandasLabeledDataProvider]):
    """
    Collection of annotated time series used in benchmarking scenarios.
    """

    def __init__(
        self,
        timeserieses: Sequence[PandasLabeledDataProvider],
    ) -> None:
        self._timeserieses = list(timeserieses)

    def __getitem__(self, index: int) -> PandasLabeledDataProvider:
        return self._timeserieses[index]

    def __len__(self) -> int:
        return len(self._timeserieses)

    @property
    def timeserieses(self) -> list[PandasLabeledDataProvider]:
        return list(self._timeserieses)

    def filter_by_annotation(self, annotation_filter: AnnotationFilter) -> "Dataset":
        filtered_timeserieses = [provider for provider in self._timeserieses if annotation_filter(provider.annotation)]
        return Dataset(filtered_timeserieses, timeseries_preprocessor=self._timeseries_preprocessor)

    def select_bisegments_by_filter(self, filter_fn: SegmentFilter | None = None) -> list[PandasLabeledDataProvider]:
        bisegments: list[PandasLabeledDataProvider] = []
        for provider in self._timeserieses:
            bisegments.extend(provider.query_bisegments(filter_fn))
        return bisegments


class DatasetLoader(Protocol):
    def load(self, path: Path, timeseries_preprocessor: TimeseriesPreprocessor | None = None) -> Dataset: ...


class RealDatasetLoader(DatasetLoader):
    @staticmethod
    def prepare_annotation(file: str | Path) -> Annotation:
        return Annotation(path=file)

    @staticmethod
    def prepare_segment_info(timeseries: pd.DataFrame) -> DatasetSegmentInfo:
        if SEGMENT_COLUMN not in timeseries.columns:
            raise ValueError(f"Timeseries must contain '{SEGMENT_COLUMN}' column")

        segments = timeseries[SEGMENT_COLUMN].to_numpy(copy=False)
        if len(segments) == 0:
            return pd.DataFrame(columns=[SEGMENT_ID_COLUMN, SEGMENT_START_COLUMN, SEGMENT_END_COLUMN])

        change_points = np.flatnonzero(segments[1:] != segments[:-1]) + 1
        starts = np.concatenate([[0], change_points])
        ends = np.concatenate([change_points - 1, [len(segments) - 1]])
        segment_ids = segments[starts]

        return pd.DataFrame(
            {
                SEGMENT_ID_COLUMN: segment_ids,
                SEGMENT_START_COLUMN: starts,
                SEGMENT_END_COLUMN: ends,
            }
        )

    @classmethod
    def load(cls, path: Path, timeseries_preprocessor: TimeseriesPreprocessor | None = None) -> Dataset:
        timeseries_files = list(path.glob("**/timeseries*.csv"))
        if not timeseries_files:
            raise FileNotFoundError(f"No timeseries files found in {path}")

        timeserieses = []
        for file in timeseries_files:
            timeseries = pd.read_csv(file)
            if timeseries_preprocessor is not None:
                timeseries = timeseries_preprocessor(timeseries)

            segment_info = cls.prepare_segment_info(timeseries)
            annotation = cls.prepare_annotation(file)
            timeserieses.append(PandasLabeledDataProvider(timeseries, segment_info=segment_info, annotation=annotation))

        return Dataset(timeserieses)
