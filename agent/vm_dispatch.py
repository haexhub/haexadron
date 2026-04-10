"""Dispatches tool calls to the BitGN VM runtime and formats results as shell-like output."""

import json
import shlex

from bitgn.vm.pcm_connect import PcmRuntimeClientSync
from bitgn.vm.pcm_pb2 import (
    AnswerRequest,
    ContextRequest,
    DeleteRequest,
    FindRequest,
    ListRequest,
    MkDirRequest,
    MoveRequest,
    Outcome,
    ReadRequest,
    SearchRequest,
    TreeRequest,
    WriteRequest,
)
from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel

from agent.models import (
    ReportTaskCompletion,
    Req_Context,
    Req_Delete,
    Req_Find,
    Req_List,
    Req_MkDir,
    Req_Move,
    Req_Read,
    Req_Search,
    Req_Tree,
    Req_Write,
)

OUTCOME_BY_NAME = {
    "OUTCOME_OK": Outcome.OUTCOME_OK,
    "OUTCOME_DENIED_SECURITY": Outcome.OUTCOME_DENIED_SECURITY,
    "OUTCOME_NONE_CLARIFICATION": Outcome.OUTCOME_NONE_CLARIFICATION,
    "OUTCOME_NONE_UNSUPPORTED": Outcome.OUTCOME_NONE_UNSUPPORTED,
    "OUTCOME_ERR_INTERNAL": Outcome.OUTCOME_ERR_INTERNAL,
}


def dispatch(vm: PcmRuntimeClientSync, cmd: BaseModel):
    if isinstance(cmd, Req_Context):
        return vm.context(ContextRequest())
    if isinstance(cmd, Req_Tree):
        return vm.tree(TreeRequest(root=cmd.root, level=cmd.level))
    if isinstance(cmd, Req_Find):
        return vm.find(
            FindRequest(
                root=cmd.root,
                name=cmd.name,
                type={"all": 0, "files": 1, "dirs": 2}[cmd.kind],
                limit=cmd.limit,
            )
        )
    if isinstance(cmd, Req_Search):
        return vm.search(
            SearchRequest(root=cmd.root, pattern=cmd.pattern, limit=cmd.limit)
        )
    if isinstance(cmd, Req_List):
        return vm.list(ListRequest(name=cmd.path))
    if isinstance(cmd, Req_Read):
        return vm.read(
            ReadRequest(
                path=cmd.path,
                number=cmd.number,
                start_line=cmd.start_line,
                end_line=cmd.end_line,
            )
        )
    if isinstance(cmd, Req_Write):
        return vm.write(
            WriteRequest(
                path=cmd.path,
                content=cmd.content,
                start_line=cmd.start_line,
                end_line=cmd.end_line,
            )
        )
    if isinstance(cmd, Req_Delete):
        return vm.delete(DeleteRequest(path=cmd.path))
    if isinstance(cmd, Req_MkDir):
        return vm.mk_dir(MkDirRequest(path=cmd.path))
    if isinstance(cmd, Req_Move):
        return vm.move(MoveRequest(from_name=cmd.from_name, to_name=cmd.to_name))
    if isinstance(cmd, ReportTaskCompletion):
        return vm.answer(
            AnswerRequest(
                message=cmd.message,
                outcome=OUTCOME_BY_NAME[cmd.outcome],
                refs=cmd.grounding_refs,
            )
        )
    raise ValueError(f"Unknown command: {cmd}")


def _render_command(command: str, body: str) -> str:
    return f"{command}\n{body}"


def _format_tree_entry(entry, prefix: str = "", is_last: bool = True) -> list[str]:
    branch = "└── " if is_last else "├── "
    lines = [f"{prefix}{branch}{entry.name}"]
    child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
    children = list(entry.children)
    for idx, child in enumerate(children):
        lines.extend(
            _format_tree_entry(child, prefix=child_prefix, is_last=idx == len(children) - 1)
        )
    return lines


def format_result(cmd: BaseModel, result) -> str:
    if result is None:
        return "{}"
    if isinstance(cmd, Req_Tree):
        root = result.root
        if not root.name:
            body = "."
        else:
            lines = [root.name]
            children = list(root.children)
            for idx, child in enumerate(children):
                lines.extend(_format_tree_entry(child, is_last=idx == len(children) - 1))
            body = "\n".join(lines)
        root_arg = cmd.root or "/"
        level_arg = f" -L {cmd.level}" if cmd.level > 0 else ""
        return _render_command(f"tree{level_arg} {root_arg}", body)
    if isinstance(cmd, Req_List):
        if not result.entries:
            body = "."
        else:
            body = "\n".join(
                f"{entry.name}/" if entry.is_dir else entry.name
                for entry in result.entries
            )
        return _render_command(f"ls {cmd.path}", body)
    if isinstance(cmd, Req_Read):
        if cmd.start_line > 0 or cmd.end_line > 0:
            start = cmd.start_line if cmd.start_line > 0 else 1
            end = cmd.end_line if cmd.end_line > 0 else "$"
            command = f"sed -n '{start},{end}p' {cmd.path}"
        elif cmd.number:
            command = f"cat -n {cmd.path}"
        else:
            command = f"cat {cmd.path}"
        return _render_command(command, result.content)
    if isinstance(cmd, Req_Search):
        root = shlex.quote(cmd.root or "/")
        pattern = shlex.quote(cmd.pattern)
        body = "\n".join(
            f"{match.path}:{match.line}:{match.line_text}" for match in result.matches
        )
        return _render_command(f"rg -n --no-heading -e {pattern} {root}", body)
    return json.dumps(MessageToDict(result), indent=2)
