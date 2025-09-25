"""Translate simplified tag queries into SQL expressions."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Mapping, Sequence

from tagger.base import TagCategory


@dataclass(frozen=True)
class QueryFragment:
    """SQL fragment describing a WHERE clause and its parameters."""

    where: str
    params: list[object]


def file_pk(alias: str) -> str:
    """Return the primary key reference for a given file table alias."""

    return f"{alias}.id"


class _TokenKind:
    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    TAG = "TAG"
    CATEGORY = "CATEGORY"
    SCORE = "SCORE"


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str


@dataclass(frozen=True)
class _Expression:
    pass


@dataclass(frozen=True)
class _TagExpr(_Expression):
    name: str


@dataclass(frozen=True)
class _CategoryExpr(_Expression):
    category: TagCategory


@dataclass(frozen=True)
class _ScoreExpr(_Expression):
    operator: str
    threshold: float


@dataclass(frozen=True)
class _UnaryExpr(_Expression):
    op: str
    operand: _Expression


@dataclass(frozen=True)
class _BinaryExpr(_Expression):
    op: str
    left: _Expression
    right: _Expression


_CATEGORY_ALIASES = {
    "general": TagCategory.GENERAL,
    "character": TagCategory.CHARACTER,
    "rating": TagCategory.RATING,
    "copyright": TagCategory.COPYRIGHT,
    "artist": TagCategory.ARTIST,
    "meta": TagCategory.META,
}

_SCORE_RE = re.compile(r"score\s*(>=|<=|=|>|<)\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _split_parentheses(token: str) -> list[str]:
    parts: list[str] = []
    buffer = []
    for char in token:
        if char in "()":
            if buffer:
                parts.append("".join(buffer))
                buffer = []
            parts.append(char)
        else:
            buffer.append(char)
    if buffer:
        parts.append("".join(buffer))
    return [part for part in parts if part]


def _tokenize(query: str) -> list[_Token]:
    raw_tokens = shlex.split(query, posix=True) if query else []
    tokens: list[_Token] = []
    for raw_token in raw_tokens:
        for raw in _split_parentheses(raw_token):
            upper = raw.upper()
            if raw == "(":
                tokens.append(_Token(_TokenKind.LPAREN, raw))
            elif raw == ")":
                tokens.append(_Token(_TokenKind.RPAREN, raw))
            elif upper == "AND":
                tokens.append(_Token(_TokenKind.AND, raw))
            elif upper == "OR":
                tokens.append(_Token(_TokenKind.OR, raw))
            elif upper == "NOT":
                tokens.append(_Token(_TokenKind.NOT, raw))
            elif raw.lower().startswith("category:"):
                name = raw.split(":", 1)[1].lower()
                if name not in _CATEGORY_ALIASES:
                    raise ValueError(f"Unknown category '{name}'")
                tokens.append(_Token(_TokenKind.CATEGORY, name))
            elif _SCORE_RE.fullmatch(raw):
                tokens.append(_Token(_TokenKind.SCORE, raw))
            else:
                tokens.append(_Token(_TokenKind.TAG, raw))
    return tokens


class _Parser:
    def __init__(self, tokens: Sequence[_Token]):
        self._tokens = tokens
        self._index = 0

    def parse(self) -> _Expression | None:
        if not self._tokens:
            return None
        expr = self._parse_or()
        if self._index != len(self._tokens):
            token = self._tokens[self._index]
            raise ValueError(f"Unexpected token '{token.value}'")
        return expr

    def _parse_or(self) -> _Expression:
        left = self._parse_and()
        while self._match(_TokenKind.OR):
            right = self._parse_and()
            left = _BinaryExpr("OR", left, right)
        return left

    def _parse_and(self) -> _Expression:
        left = self._parse_not()
        while True:
            if self._match(_TokenKind.AND):
                right = self._parse_not()
                left = _BinaryExpr("AND", left, right)
                continue
            if self._peek_is_operand():
                right = self._parse_not()
                left = _BinaryExpr("AND", left, right)
                continue
            break
        return left

    def _parse_not(self) -> _Expression:
        if self._match(_TokenKind.NOT):
            operand = self._parse_not()
            return _UnaryExpr("NOT", operand)
        return self._parse_primary()

    def _parse_primary(self) -> _Expression:
        if self._match(_TokenKind.LPAREN):
            expr = self._parse_or()
            if not self._match(_TokenKind.RPAREN):
                raise ValueError("Missing closing parenthesis")
            return expr

        token = self._advance()
        if token is None:
            raise ValueError("Unexpected end of query")
        if token.kind == _TokenKind.TAG:
            return _TagExpr(name=token.value)
        if token.kind == _TokenKind.CATEGORY:
            category = _CATEGORY_ALIASES[token.value]
            return _CategoryExpr(category=category)
        if token.kind == _TokenKind.SCORE:
            match = _SCORE_RE.fullmatch(token.value)
            if match is None:
                raise ValueError(f"Invalid score predicate '{token.value}'")
            operator, threshold_s = match.groups()
            threshold = float(threshold_s)
            return _ScoreExpr(operator=operator, threshold=threshold)
        raise ValueError(f"Unsupported token '{token.value}'")

    def _match(self, kind: str) -> bool:
        if self._peek(kind):
            self._index += 1
            return True
        return False

    def _peek(self, kind: str) -> bool:
        token = self._current()
        return token is not None and token.kind == kind

    def _current(self) -> _Token | None:
        if self._index >= len(self._tokens):
            return None
        return self._tokens[self._index]

    def _advance(self) -> _Token | None:
        token = self._current()
        if token is not None:
            self._index += 1
        return token

    def _peek_is_operand(self) -> bool:
        token = self._current()
        if token is None:
            return False
        return token.kind in {
            _TokenKind.TAG,
            _TokenKind.CATEGORY,
            _TokenKind.SCORE,
            _TokenKind.LPAREN,
            _TokenKind.NOT,
        }


_FALLBACK_THRESHOLDS: dict[int, float] = {
    0: 0.35,
    1: 0.25,
    3: 0.25,
    -1: 0.0,
}


def _normalize_thresholds(
    thresholds: Mapping[int, float] | None,
) -> dict[int, float]:
    mapping = dict(_FALLBACK_THRESHOLDS)
    if thresholds:
        for key, value in thresholds.items():
            try:
                mapping[int(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return mapping


def _threshold_tuple(
    thresholds: Mapping[int, float] | None,
) -> tuple[float, float, float, float]:
    normalized = _normalize_thresholds(thresholds)
    return (
        normalized.get(0, 0.0),
        normalized.get(1, 0.0),
        normalized.get(3, 0.0),
        normalized.get(-1, 0.0),
    )


def _build_sql(
    expr: _Expression | None,
    *,
    file_alias: str = "f",
    thresholds: Mapping[int, float] | None = None,
) -> QueryFragment:
    if expr is None:
        return QueryFragment(where="1=1", params=[])

    where, params = _compile_expression(
        expr,
        file_alias=file_alias,
        thresholds=thresholds,
    )
    return QueryFragment(where=where, params=params)


def _compile_expression(
    expr: _Expression,
    *,
    file_alias: str,
    thresholds: Mapping[int, float] | None,
) -> tuple[str, list[object]]:
    if isinstance(expr, _TagExpr):
        general_thr, character_thr, copyright_thr, default_thr = _threshold_tuple(
            thresholds
        )
        clause = (
            "EXISTS ("
            "SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
            f"WHERE ft.file_id = {file_pk(file_alias)} "
            "AND t.name = ? "
            "AND ft.score >= CASE t.category "
            "WHEN 0 THEN ? "
            "WHEN 1 THEN ? "
            "WHEN 3 THEN ? "
            "ELSE ? "
            "END)"
        )
        return (
            clause,
            [
                expr.name,
                general_thr,
                character_thr,
                copyright_thr,
                default_thr,
            ],
        )
    if isinstance(expr, _CategoryExpr):
        normalized = _normalize_thresholds(thresholds)
        category_key = int(expr.category)
        threshold_value = float(normalized.get(category_key, 0.0))
        clause = (
            "EXISTS ("
            "SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
            f"WHERE ft.file_id = {file_pk(file_alias)} "
            "AND t.category = ? "
            "AND ft.score >= ?)"
        )
        return clause, [category_key, threshold_value]
    if isinstance(expr, _ScoreExpr):
        clause = (
            "EXISTS (SELECT 1 FROM file_tags ft "
            f"WHERE ft.file_id = {file_pk(file_alias)} AND ft.score {expr.operator} ? )"
        )
        return clause, [expr.threshold]
    if isinstance(expr, _UnaryExpr):
        inner, params = _compile_expression(
            expr.operand,
            file_alias=file_alias,
            thresholds=thresholds,
        )
        return f"NOT ({inner})", params
    if isinstance(expr, _BinaryExpr):
        left_sql, left_params = _compile_expression(
            expr.left,
            file_alias=file_alias,
            thresholds=thresholds,
        )
        right_sql, right_params = _compile_expression(
            expr.right,
            file_alias=file_alias,
            thresholds=thresholds,
        )
        combined = f"({left_sql}) {expr.op} ({right_sql})"
        return combined, left_params + right_params

    raise TypeError(f"Unhandled expression type: {expr}")


def translate_query(
    query: str,
    *,
    file_alias: str = "f",
    thresholds: Mapping[int, float] | None = None,
) -> QueryFragment:
    """Convert a simplified tag query string into an SQL WHERE fragment."""
    tokens = _tokenize(query)
    parser = _Parser(tokens)
    expr = parser.parse()
    return _build_sql(expr, file_alias=file_alias, thresholds=thresholds)


__all__ = ["QueryFragment", "file_pk", "translate_query"]
