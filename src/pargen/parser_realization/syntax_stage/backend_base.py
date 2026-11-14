from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from ..backend_contract import BackendAdapterContract, backend_adapter_contract
from .schema import SyntaxResult


class SyntaxBackend(ABC):

    name: str

    def backend_contract(self) -> BackendAdapterContract:
        return backend_adapter_contract(self.name)

    @abstractmethod
    def check(
        self,
        *,
        plugin_src_dir: Path,
        protocol_filter_name: str,
        file_prefix: str,
        logs_root: Path,
        backend_config: Optional[Dict[str, Any]] = None,
    ) -> SyntaxResult:
        raise NotImplementedError


    def extra_llm_rules(
        self,
        *,
        protocol_filter_name: str,
        file_prefix: str,
        backend_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        return ""
