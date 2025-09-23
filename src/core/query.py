"""Translate simplified tag queries into SQL expressions."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Sequence

from tagger.base import TagCategory


@dataclass(frozen=True)
class QueryFragment:
    """SQL fragment describing a WHERE clause and its parameters."""

    where: str
    params: list[object]


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


def _build_sql(expr: _Expression) -> QueryFragment:
    if expr is None:
        return QueryFragment(where="1=1", params=[])

    where, params = _compile_expression(expr)
    return QueryFragment(where=where, params=params)


def _compile_expression(expr: _Expression) -> tuple[str, list[object]]:
    if isinstance(expr, _TagExpr):
        clause = (
            "EXISTS ("
            "SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
            "WHERE ft.file_id = files.id AND t.name = ?)"
        )
        return clause, [expr.name]
    if isinstance(expr, _CategoryExpr):
        clause = (
            "EXISTS ("
            "SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
            "WHERE ft.file_id = files.id AND t.category = ?)"
        )
        return clause, [int(expr.category)]
    if isinstance(expr, _ScoreExpr):
        clause = "EXISTS (SELECT 1 FROM file_tags ft WHERE ft.file_id = files.id " f"AND ft.score {expr.operator} ? )"
        return clause, [expr.threshold]
    if isinstance(expr, _UnaryExpr):
        inner, params = _compile_expression(expr.operand)
        return f"NOT ({inner})", params
    if isinstance(expr, _BinaryExpr):
        left_sql, left_params = _compile_expression(expr.left)
        right_sql, right_params = _compile_expression(expr.right)
        combined = f"({left_sql}) {expr.op} ({right_sql})"
        return combined, left_params + right_params

    raise TypeError(f"Unhandled expression type: {expr}")


def translate_query(query: str) -> QueryFragment:
    """Convert a simplified tag query string into an SQL WHERE fragment."""
    tokens = _tokenize(query)
    parser = _Parser(tokens)
    expr = parser.parse()
    return _build_sql(expr)


__all__ = ["QueryFragment", "translate_query"]
