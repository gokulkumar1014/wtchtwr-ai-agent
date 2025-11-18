from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

class TablePayload(BaseModel):
    name: str = 'Result'
    columns: List[str] = Field(default_factory=list)
    data: List[dict] = Field(default_factory=list)
    source: Optional[str] = None
    sql: Optional[str] = None

class MessagePayload(BaseModel):
    sql: Optional[str] = None
    params: Optional[List[str]] = None
    tables: List[TablePayload] = Field(default_factory=list)
    summary: str = Field(default='')
    question: Optional[str] = None
    export: Optional[Dict[str, object]] = None
    template_id: Optional[str] = None
    row_count: Optional[int] = None
    duration_ms: Optional[float] = None
    response_type: str = Field(default='text')
    action: Optional[str] = None
    targets: Optional[Dict[str, str]] = None

    def model_post_init(self, __context: Any) -> None:
        if self.tables is None:
            self.tables = []

class Message(BaseModel):
    id: str
    role: str
    content: Optional[str] = None
    nl_summary: Optional[str] = None
    payload: Optional[Any] = None
    timestamp: str

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
        json_encoders = {dict: lambda v: v, list: lambda v: v}

class Conversation(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    latency: Optional[int] = None

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True
        json_encoders = {dict: lambda v: v, list: lambda v: v}

class ConversationCreate(BaseModel):
    title: Optional[str] = None

class MessageCreate(BaseModel):
    role: str
    content: str

class ConversationSummary(BaseModel):
    conversation_id: str
    concise: str
    detailed: str

class ExportRequest(BaseModel):
    table_index: int = 0
    delivery: Literal["download", "email"] = "download"
    email: Optional[str] = None
    email_mode: Literal["sql", "csv", "both"] = "csv"

class ExportResponse(BaseModel):
    token: str
    format: str
    rows: int
    expires_at: str
    filename: str
    session_only: bool = True

class EmailSummaryRequest(BaseModel):
    email: str
    variant: Literal["concise", "detailed"] = "concise"


class ExportActionResponse(BaseModel):
    delivery: Literal['download', 'email']
    metadata: Optional[ExportResponse] = None
    detail: Optional[str] = None

class DataExplorerColumn(BaseModel):
    name: str
    data_type: str
    description: Optional[str] = None

class DataExplorerTable(BaseModel):
    name: str
    columns: List[DataExplorerColumn]

class DataExplorerSchemaResponse(BaseModel):
    tables: List[DataExplorerTable]

class DataExplorerColumnSelection(BaseModel):
    table: str
    column: str

class DataExplorerJoin(BaseModel):
    left_table: str
    left_column: str
    right_table: str
    right_column: str

class DataExplorerFilter(BaseModel):
    table: str
    column: str
    operator: str = 'equals'
    value: Optional[Any] = None

class DataExplorerSort(BaseModel):
    table: str
    column: str
    direction: Literal['asc', 'desc'] = 'asc'

class DataExplorerQueryRequest(BaseModel):
    tables: List[str]
    columns: List[DataExplorerColumnSelection]
    filters: Optional[List[DataExplorerFilter]] = None
    sort: Optional[List[DataExplorerSort]] = None
    joins: Optional[List[DataExplorerJoin]] = None
    limit: Optional[int] = None

class DataExplorerQueryResponse(BaseModel):
    sql: str
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    limit: int
    tables: List[str]

class DataExplorerExportRequest(DataExplorerQueryRequest):
    delivery: Literal['download', 'email'] = 'download'
    email: Optional[str] = None
