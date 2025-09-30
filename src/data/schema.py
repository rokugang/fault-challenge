from __future__ import annotations

from typing import Final, List

import pandera.pandas as pa
from pandera.typing import Series

REFERENCE_BASE_SCHEMA: Final[pa.DataFrameSchema] = pa.DataFrameSchema(
    {
        "Temperatura do ar ambiente": pa.Column(dtype=None, nullable=True, required=False),
        "Rotação do motor - RPM": pa.Column(dtype=None, nullable=True, required=False),
        "Carga calculada do motor": pa.Column(dtype=None, nullable=True, required=False),
        "Ajuste de combustível de longo prazo - Banco 1": pa.Column(dtype=None, nullable=True, required=False),
        "Temperatura do ar admitido - ACT": pa.Column(dtype=None, nullable=True, required=False),
        "Temperatura do líquido de arrefecimento do motor - CTS": pa.Column(dtype=None, nullable=True, required=False),
        "Ajuste de combustível de curto prazo - Banco 1": pa.Column(dtype=None, nullable=True, required=False),
        "Sonda lambda - Banco 1, sensor 2": pa.Column(dtype=None, nullable=True, required=False),
        "Pressão barométrica": pa.Column(dtype=None, nullable=True, required=False),
        "Tensão do módulo": pa.Column(dtype=None, nullable=True, required=False),
        "Posição relativa da borboleta - TPS": pa.Column(dtype=None, nullable=True, required=False),
        "Posição absoluta da borboleta - TPS": pa.Column(dtype=None, nullable=True, required=False),
        "Posição absoluta da borboleta - Sensor B": pa.Column(dtype=None, nullable=True, required=False),
        "Sonda lambda - Banco 1, sensor 1": pa.Column(dtype=None, nullable=True, required=False),
        "Ajuste de combustível de curto prazo - Banco 1, sensor 1": pa.Column(dtype=None, nullable=True, required=False),
        "Pressão no coletor de admissão - MAP": pa.Column(dtype=None, nullable=True, required=False),
        "Estado do sistema de combustível": pa.Column(dtype=None, nullable=True, required=False),
        "Carga absoluta do motor": pa.Column(dtype=None, nullable=True, required=False),
        "Nº de falhas na memória": pa.Column(dtype=None, nullable=True, required=False),
        "Altitude": pa.Column(dtype=None, nullable=True, required=False),
    },
    coerce=False,
    add_missing_columns=True,
)

MANDATORY_CONTEXT_COLUMNS: Final[List[str]] = [
    "Temperatura do líquido de arrefecimento do motor - CTS",
    "Carga calculada do motor",
    "Rotação do motor - RPM",
    "Altitude",
]

NUMERIC_COLUMNS: Final[List[str]] = [
    "Temperatura do ar ambiente",
    "Rotação do motor - RPM",
    "Carga calculada do motor",
    "Ajuste de combustível de longo prazo - Banco 1",
    "Temperatura do ar admitido - ACT",
    "Temperatura do líquido de arrefecimento do motor - CTS",
    "Ajuste de combustível de curto prazo - Banco 1",
    "Sonda lambda - Banco 1, sensor 2",
    "Pressão barométrica",
    "Tensão do módulo",
    "Posição relativa da borboleta - TPS",
    "Posição absoluta da borboleta - TPS",
    "Posição absoluta da borboleta - Sensor B",
    "Sonda lambda - Banco 1, sensor 1",
    "Ajuste de combustível de curto prazo - Banco 1, sensor 1",
    "Pressão no coletor de admissão - MAP",
    "Carga absoluta do motor",
    "Nº de falhas na memória",
    "Altitude",
]
