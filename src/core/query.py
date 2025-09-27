"""Utilities for parsing whitespace-delimited tag queries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from tagger.base import TagCategory


@dataclass(frozen=True)
class QueryFragment:
    """SQL fragment describing a WHERE clause and its parameters."""

    where: str
    params: list[object]


@dataclass(frozen=True)
class TagSpec:
    """Description of a single tag token extracted from a query."""

    raw: str
    category: Optional[str]
    name: str
    negative: bool = False


_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)

TAG_CHARS = r"[A-Za-z0-9:_'\-]+"
_TAG_VALID_RE = re.compile(f"^{TAG_CHARS}$")


class _TokenKind:
    """Token kinds produced by the boolean lexer."""

    LPAREN = "LPAREN"
    RPAREN = "RPAREN"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    TAG = "TAG"


@dataclass(frozen=True)
class _Token:
    """Lexical token containing its kind and original value."""

    kind: str
    value: str


@dataclass(frozen=True)
class _Expression:
    """Base class for parsed boolean expressions."""


@dataclass(frozen=True)
class _TagExpr(_Expression):
    """Leaf expression describing a single tag specification."""

    spec: TagSpec


@dataclass(frozen=True)
class _UnaryExpr(_Expression):
    """Unary boolean operation such as NOT."""

    op: str
    operand: _Expression


@dataclass(frozen=True)
class _BinaryExpr(_Expression):
    """Binary boolean operation such as AND/OR."""

    op: str
    left: _Expression
    right: _Expression

_CATEGORY_ALIAS_TO_CANONICAL: Dict[str, str] = {
    "0": "general",
    "general": "general",
    "1": "character",
    "character": "character",
    "2": "rating",
    "rating": "rating",
    "3": "copyright",
    "copyright": "copyright",
    "4": "artist",
    "artist": "artist",
    "5": "meta",
    "meta": "meta",
}

_CANONICAL_TO_CATEGORY: Dict[str, TagCategory] = {
    "general": TagCategory.GENERAL,
    "character": TagCategory.CHARACTER,
    "rating": TagCategory.RATING,
    "copyright": TagCategory.COPYRIGHT,
    "artist": TagCategory.ARTIST,
    "meta": TagCategory.META,
}


def tokenize_whitespace_only(query: str) -> List[str]:
    """Split ``query`` by Unicode whitespace without special quoting rules."""

    if not query:
        return []
    normalized = query.replace("\u3000", " ")
    return [token for token in _WHITESPACE_RE.split(normalized) if token]


def strip_negative_prefix(token: str) -> Tuple[bool, str]:
    """Return a tuple describing whether ``token`` is negative and its core value."""

    if token.startswith("-") and len(token) > 1:
        return True, token[1:]
    return False, token


def is_not_keyword(token: str) -> bool:
    """Return ``True`` if ``token`` is the NOT keyword (case insensitive)."""

    return token.upper() == "NOT"


def split_category(raw: str) -> Tuple[Optional[str], str]:
    """Split ``raw`` into an optional category and the remaining tag name."""

    if ":" not in raw:
        return None, raw
    left, right = raw.split(":", 1)
    if not left or not right:
        return None, raw
    canonical = _CATEGORY_ALIAS_TO_CANONICAL.get(left.lower())
    if canonical is None:
        return None, raw
    return canonical, right


def parse_search(query: str) -> Dict[str, List[TagSpec]]:
    """Parse ``query`` into positive and negative tag specifications."""

    tokens = tokenize_whitespace_only(query)
    include: List[TagSpec] = []
    exclude: List[TagSpec] = []
    free: List[TagSpec] = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if is_not_keyword(token):
            if i + 1 < len(tokens):
                candidate = tokens[i + 1]
                _, core = strip_negative_prefix(candidate)
                category, name = split_category(core)
                spec = TagSpec(raw=core, category=category, name=name, negative=True)
                exclude.append(spec)
                i += 2
                continue
            free.append(TagSpec(raw=token, category=None, name=token))
            i += 1
            continue

        is_negative, core = strip_negative_prefix(token)
        if not core:
            free.append(TagSpec(raw=token, category=None, name=token, negative=is_negative))
            i += 1
            continue
        category, name = split_category(core)
        spec = TagSpec(raw=core, category=category, name=name, negative=is_negative)
        if is_negative:
            exclude.append(spec)
        else:
            include.append(spec)
        i += 1

    return {"include": include, "exclude": exclude, "free": free}


def _split_parens_for_lex(raw: str, balance: int) -> Tuple[List[str], int]:
    """Split ``raw`` while tracking parenthesis ``balance`` for grouping."""

    if "(" in raw and ")" in raw and not any(ch.isspace() for ch in raw):
        return [raw], balance

    tokens: List[str] = []
    buffer: List[str] = []
    i = 0
    while i < len(raw):
        char = raw[i]
        if char == "\\" and i + 1 < len(raw) and raw[i + 1] in "()":
            buffer.append(char)
            buffer.append(raw[i + 1])
            i += 2
            continue
        if char == "(":
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            tokens.append("(")
            balance += 1
            i += 1
            continue
        if char == ")":
            if balance == 0:
                buffer.append(char)
                i += 1
                continue
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            tokens.append(")")
            balance = max(balance - 1, 0)
            i += 1
            continue
        buffer.append(char)
        i += 1
    if buffer:
        tokens.append("".join(buffer))
    return [token for token in tokens if token], balance


def _make_tag_spec(raw: str) -> TagSpec:
    """Return a TagSpec for ``raw`` while preserving its original text."""

    is_negative, core = strip_negative_prefix(raw)
    category, name = split_category(core)
    return TagSpec(raw=core, category=category, name=name, negative=is_negative)


def _normalise_spec_for_expr(spec: TagSpec) -> TagSpec:
    """Return a positive TagSpec suitable for embedding in expressions."""

    if not spec.negative:
        return spec
    return TagSpec(raw=spec.raw, category=spec.category, name=spec.name, negative=False)


def _lex_boolean_tokens(query: str) -> List[_Token]:
    """Lex ``query`` into boolean-aware tokens without raising errors."""

    tokens: List[_Token] = []
    balance = 0
    for token in tokenize_whitespace_only(query):
        pieces, balance = _split_parens_for_lex(token, balance)
        for piece in pieces:
            if piece == "(":
                tokens.append(_Token(_TokenKind.LPAREN, piece))
            elif piece == ")":
                tokens.append(_Token(_TokenKind.RPAREN, piece))
            else:
                upper = piece.upper()
                if piece == upper and upper in {_TokenKind.AND, _TokenKind.OR, _TokenKind.NOT}:
                    tokens.append(_Token(upper, piece))
                else:
                    tokens.append(_Token(_TokenKind.TAG, piece))
    return tokens


class _Parser:
    """Recursive-descent parser for boolean tag expressions."""

    def __init__(self, tokens: Sequence[_Token]):
        self._tokens = tokens
        self._index = 0

    def parse(self) -> Optional[_Expression]:
        if not self._tokens:
            return None
        expr = self._parse_or()
        if self._index != len(self._tokens):
            return None
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
                return None
            return expr

        token = self._advance()
        if token is None:
            return None
        if token.kind != _TokenKind.TAG:
            return None
        spec = _make_tag_spec(token.value)
        base_spec = _normalise_spec_for_expr(spec)
        expr: _Expression = _TagExpr(spec=base_spec)
        if spec.negative:
            expr = _UnaryExpr("NOT", expr)
        return expr

    def _match(self, kind: str) -> bool:
        if self._peek(kind):
            self._index += 1
            return True
        return False

    def _peek(self, kind: str) -> bool:
        token = self._current()
        return token is not None and token.kind == kind

    def _current(self) -> Optional[_Token]:
        if self._index >= len(self._tokens):
            return None
        return self._tokens[self._index]

    def _advance(self) -> Optional[_Token]:
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
            _TokenKind.LPAREN,
            _TokenKind.NOT,
        }


def _resolve_category_key(category: Optional[str]) -> Optional[int]:
    """Convert a canonical category label into its database key."""

    if category is None:
        return None
    lookup = _CANONICAL_TO_CATEGORY.get(category.lower())
    if lookup is None:
        return None
    return int(lookup)


def _exists_clause_for_tag(alias_file: str, spec: TagSpec) -> Tuple[str, List[object]]:
    """Return an EXISTS clause and parameters for ``spec`` against ``alias_file``."""

    category_key = _resolve_category_key(spec.category)
    if category_key is not None:
        clause = (
            "EXISTS (SELECT 1 FROM file_tags ft "
            "JOIN tags t ON t.id = ft.tag_id "
            f"WHERE ft.file_id = {alias_file}.id "
            "AND t.category = ? AND t.name = ?)"
        )
        return clause, [category_key, spec.name]
    clause = (
        "EXISTS (SELECT 1 FROM file_tags ft "
        "JOIN tags t ON t.id = ft.tag_id "
        f"WHERE ft.file_id = {alias_file}.id "
        "AND t.name = ?)"
    )
    return clause, [spec.name]


def _compile_expression(expr: _Expression, *, file_alias: str) -> Tuple[str, List[object], int]:
    """Compile an expression tree into SQL, parameters and precedence."""

    if isinstance(expr, _TagExpr):
        clause, params = _exists_clause_for_tag(file_alias, expr.spec)
        return clause, params, 3
    if isinstance(expr, _UnaryExpr) and expr.op == "NOT":
        inner_sql, inner_params, inner_prec = _compile_expression(expr.operand, file_alias=file_alias)
        if inner_prec < 2:
            inner_sql = f"({inner_sql})"
        return f"NOT {inner_sql}", inner_params, 2
    if isinstance(expr, _BinaryExpr):
        current_prec = 1 if expr.op == "AND" else 0
        left_sql, left_params, left_prec = _compile_expression(expr.left, file_alias=file_alias)
        right_sql, right_params, right_prec = _compile_expression(expr.right, file_alias=file_alias)
        if left_prec < current_prec:
            left_sql = f"({left_sql})"
        if right_prec < current_prec:
            right_sql = f"({right_sql})"
        combined_params = left_params + right_params
        return f"{left_sql} {expr.op} {right_sql}", combined_params, current_prec
    raise TypeError(f"Unsupported expression node: {expr}")


def translate_query(query: str, *, file_alias: str = "f") -> QueryFragment:
    """Convert ``query`` into a QueryFragment for database filtering."""

    tokens = _lex_boolean_tokens(query)
    parser = _Parser(tokens)
    expr = parser.parse()
    if expr is None:
        return QueryFragment(where="1=1", params=[])
    try:
        where, params, _ = _compile_expression(expr, file_alias=file_alias)
    except TypeError:
        return QueryFragment(where="1=1", params=[])
    return QueryFragment(where=where, params=params)


def extract_positive_tag_terms(query: str) -> List[str]:
    """Return lower-cased unique tag names that appear positively in ``query``."""

    tokens = _lex_boolean_tokens(query)
    parser = _Parser(tokens)
    expr = parser.parse()
    if expr is None:
        return []

    seen: set[str] = set()
    ordered: List[str] = []

    def walk(node: _Expression, negated: bool = False) -> None:
        if isinstance(node, _TagExpr):
            if negated:
                return
            name = node.spec.name.strip()
            if not name or name.endswith(":"):
                return
            if _TAG_VALID_RE.fullmatch(name) is None:
                return
            lowered = name.lower()
            if lowered not in seen:
                seen.add(lowered)
                ordered.append(lowered)
            return
        if isinstance(node, _UnaryExpr) and node.op == "NOT":
            walk(node.operand, not negated)
            return
        if isinstance(node, _BinaryExpr):
            walk(node.left, negated)
            walk(node.right, negated)

    walk(expr)
    return ordered


__all__ = [
    "QueryFragment",
    "TAG_CHARS",
    "TagSpec",
    "extract_positive_tag_terms",
    "parse_search",
    "tokenize_whitespace_only",
    "translate_query",
]
