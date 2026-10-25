from __future__ import annotations

import asyncio
import json
import logging
import re
from enum import StrEnum
from collections.abc import Collection
from typing import Annotated, Any, Protocol

from pydantic import BaseModel, StringConstraints, ValidationError, field_validator

from app.llm import FAILOVER_ERRORS, AssistantTurn, Chat, LLMClient, LLMError, TokenUsage, ToolCall
from app.retrieval.base import Retriever
from app.retrieval.titles import TitleIndex
from app.schemas import AskResponse, Source, SourceType
from app.sources.superhero import Hero, SuperheroAPIError

logger = logging.getLogger(__name__)


class ToolName(StrEnum):
    SEARCH_DOCUMENTS = "search_documents"
    GET_SUPERHERO = "get_superhero"


_KNOWN_TOOLS = frozenset(ToolName)


def build_system_prompt(collections: dict[str, str]) -> str:
    """The system prompt, listing the document collections so new datasets need no prompt edits."""
    listed = "\n".join(f"- {name}: {description}" for name, description in collections.items())
    return f"""\
You answer questions using only your tools and these sources:
Document collections ({ToolName.SEARCH_DOCUMENTS}):
{listed}
Superhero database ({ToolName.GET_SUPERHERO}): comic-book characters.

Rules:
1. Use the source(s) the question needs. A tool result may note that a name also exists in another \
source: unless the question clearly limits itself to one, check both and cover both.
2. Search once per figure; one search rarely covers two.
3. Answer only from tool results. If they don't contain the answer, say you don't know; if they contradict \
the question's premise, say so.
4. Cite the [S#] id of each fact right after it, e.g. "Thor wields Mjölnir [S2]."
5. For questions outside these sources' topics, call no tools and say what you cover.
6. Your final message must hold the whole answer: text sent alongside tool calls is not shown.
7. Be concise."""


def build_tools(collections: dict[str, str]) -> list[dict[str, Any]]:
    """OpenAI-format tool definitions; the collection argument is limited to the loaded collections."""
    return [
        {
            "type": "function",
            "function": {
                "name": ToolName.SEARCH_DOCUMENTS,
                "description": "Keyword search over the document collections. Name the figure explicitly; "
                               "one call per figure.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Keywords, e.g. 'Thor hammer Mjolnir'."},
                        "collection": {"type": "string", "enum": sorted(collections),
                                       "description": "Search one collection only; omit to search all."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": ToolName.GET_SUPERHERO,
                "description": "Look up one comic-book character by name: publisher, first appearance, alignment, "
                               "powerstats, appearance, affiliations. It cannot list or rank characters, so "
                               "'who is the smartest/strongest' questions can't be answered from it.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "Character name, e.g. 'Wonder Woman'."}},
                    "required": ["name"],
                },
            },
        },
    ]


NO_ANSWER = "Sorry, I couldn't produce an answer."
TOOL_BUDGET_EXHAUSTED = "ERROR: tool budget exhausted."
CITATION_REPAIR = """\
These sentences in your answer have no [S#] citation:
{sentences}

Rewrite the answer so every factual sentence cites the [S#] id(s) of the tool result it comes from, \
and remove any statement no tool result supports. Keep statements saying something was not found or \
is unavailable, and everything else, unchanged. Reply with the full corrected answer only."""
DUPLICATE_CALL = "You already made this exact call; its result is above. Use it, or try a different query or tool."
# Cross-source hints: the agent notices a name exists in a source the model hasn't used, and says so.
SUPERHERO_HINT = ("Note: {names} also exist(s) in the superhero database ({tool}). If the question could be "
                  "about the comic-book character, look it up too.")
DOCUMENT_HINT = ("Note: '{name}' also appears in the '{collections}' document collection ({tool}). If the "
                 "question could be about that, search it too.")
MAX_HINT_LOOKUPS = 3  # names per search worth checking against the superhero database (looked up in parallel)
NO_PASSAGES = "No matching passages found."
SUPERHERO_UNAVAILABLE = "ERROR: the Superhero API is unavailable right now. Tell the user superhero data could not be retrieved."
_CITATION_RE = re.compile(r"\bS\d+\b")
# Models drift from the requested [S1] form, e.g. gpt-oss writes 【S1】, 【S5†L9-L12】 or [ S1 ].
_LOOSE_CITATION_RE = re.compile(r"[【\[]\s*(S\d+)(?:†[^】\]]*)?\s*[】\]]")
# Splitting an answer into checkable sentences: list markers, then sentence ends before any capital (Æsir too).
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[^\W\d_])")
# Sentences saying the sources don't cover something have nothing to cite. Phrases are tied to the speaker or
# the sources so that ordinary claims ("Achilles cannot be harmed...") are still checked.
_NO_ANSWER_RE = re.compile(
    r"(?:don't|do not) know|not (?:found|mentioned|available|specified|provided|listed|given|included)|"
    r"\bno (?:information|record|data|mention|details)\b|"
    r"(?:don't|do not|doesn't|does not|didn't) (?:contain|mention|say|give|provide|include|have)|"
    r"\bI (?:can't|cannot|am unable|am not able)\b|\bI'm (?:unable|not able|sorry)\b|unable to (?:find|retrieve|answer)|"
    r"only (?:answer|cover)|\bsorry\b|could not be (?:retrieved|found)|\bunavailable\b|"
    r"\bno (?:superhero|character|entry|article|god)\b.*\b(?:found|named)\b|\bno\b[^.]*\b(?:was|were) found\b",
    re.IGNORECASE,
)
MIN_FACT_WORDS = 6  # shorter fragments ("In mythology:", "Yes.") aren't claims worth citing
# Invisible or typographic characters some models emit; they break copy-paste and string matching.
_CHARACTER_FIXES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u200b": None, "\u200c": None, "\u200d": None})


class SearchArgs(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    collection: str | None = None

    @field_validator("collection", mode="before")
    @classmethod
    def _blank_means_all(cls, value: str | None) -> str | None:
        """Treat an empty collection name as "search every collection"."""
        return value or None


class HeroArgs(BaseModel):
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class HeroSearch(Protocol):
    """What the agent needs from a superhero source."""

    async def search(self, name: str) -> list[Hero]:
        """Return heroes matching name; raise SuperheroAPIError if the source is unavailable."""
        ...


class _RequestState:
    """Sources and warnings collected while answering one question."""

    def __init__(self) -> None:
        """Start with no sources, tools or warnings."""
        self.sources: dict[str, Source] = {}
        self.tools_used: list[str] = []
        self.warnings: list[str] = []
        self.usage = TokenUsage()
        self.llm_calls = 0
        self._seen_calls: set[tuple[str, str]] = set()

    def add_source(self, source_type: SourceType, title: str, url: str | None, collection: str | None = None) -> str:
        """Register a source and return the id ("S1", "S2", ...) the model should cite it by.

        Passages from the same section share one id, so the response never lists the same source twice.
        """
        for source in self.sources.values():
            if (source.type, source.title, source.url, source.collection) == (source_type, title, url, collection):
                return source.id
        source_id = f"S{len(self.sources) + 1}"
        self.sources[source_id] = Source(id=source_id, type=source_type, title=title, url=url, collection=collection)
        return source_id

    def mark_used(self, calls: list[ToolCall]) -> None:
        """Record the tools of a turn before running any, so a hint never points to a source being queried.

        Unknown tool names (a model inventing or misspelling one) are not recorded as used.
        """
        for call in calls:
            if call.name in _KNOWN_TOOLS and call.name not in self.tools_used:
                self.tools_used.append(call.name)

    def is_first_call(self, call: ToolCall) -> bool:
        """Record a tool call; False if the same tool was already called with the same arguments.

        Models sometimes repeat an identical call that came back empty, burning tool rounds.
        """
        key = (call.name, " ".join(call.arguments.lower().split()))
        if key in self._seen_calls:
            return False
        self._seen_calls.add(key)
        return True

    def cited_sources(self, answer: str) -> list[Source]:
        """Sources cited in the answer, in order of first citation."""
        # Only ids we issued count, so the model can't invent a source.
        cited_ids = dict.fromkeys(i for i in _CITATION_RE.findall(answer) if i in self.sources)
        return [self.sources[source_id] for source_id in cited_ids]


class Agent:
    """Answers a question by letting the LLM call retrieval tools, then attributes the answer to sources."""

    def __init__(self, llm: LLMClient, retriever: Retriever, superheroes: HeroSearch, *,
                 collections: dict[str, str], titles: TitleIndex, top_k: int = 4, max_tool_rounds: int = 3):
        """Create an agent.

        Args:
            llm: Chat client used for routing and answering.
            retriever: Passage retriever over all document collections.
            superheroes: Superhero lookup source.
            collections: Description of each document collection, by name.
            titles: Article names per collection, used for cross-source hints.
            top_k: Passages returned per document search.
            max_tool_rounds: Model turns allowed to call tools before an answer is forced.
        """
        self._llm = llm
        self._retriever = retriever
        self._superheroes = superheroes
        self._collections = collections
        self._titles = titles
        self._top_k = top_k
        self._max_tool_rounds = max_tool_rounds
        self._system_prompt = build_system_prompt(collections)
        self._tools = build_tools(collections)

    async def ask(self, question: str) -> AskResponse:
        """Answer a question with sources.

        Raises:
            LLMError: If the LLM can't be reached or rejects the request; tool failures don't raise.
        """
        turn, state = await self._llm.run_conversation(lambda chat: self._converse(chat, question))
        logger.info("Answered with %d LLM calls: %d prompt tokens (%d cached), %d completion tokens",
                    state.llm_calls, state.usage.prompt, state.usage.cached_prompt, state.usage.completion)
        answer = _clean_answer(turn.content or "") or NO_ANSWER
        if state.sources and (still_uncited := uncited_sentences(answer, state.sources)):
            state.warnings.append(f"{len(still_uncited)} statement(s) in the answer are not linked to a source.")
        return AskResponse(
            answer=answer,
            sources=_attribute(answer, state, turn.model),
            tools_used=state.tools_used,
            warnings=state.warnings,
        )

    async def _converse(self, chat: Chat, question: str) -> tuple[AssistantTurn, _RequestState]:
        """Run the tool-calling loop from scratch on one provider.

        State is created here, not in ask(), so a provider failover restarts with no sources
        left over from the abandoned attempt.

        Returns:
            The model's final turn (no tool calls) and the sources/warnings collected on the way.
        """
        state = _RequestState()

        async def counted_chat(*args: Any, **kwargs: Any) -> AssistantTurn:
            """Call the model and add the turn's token usage to this request's totals."""
            turn = await chat(*args, **kwargs)
            state.usage += turn.usage
            state.llm_calls += 1
            return turn

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": question},
        ]
        turn = await counted_chat(messages, tools=self._tools)
        rounds = 0
        while turn.tool_calls and rounds < self._max_tool_rounds:
            state.mark_used(turn.tool_calls)
            results = await asyncio.gather(*(self._run_tool(call, state) for call in turn.tool_calls))
            messages += _tool_exchange(turn, results)
            turn = await counted_chat(messages, tools=self._tools)
            rounds += 1

        if turn.tool_calls:
            # Out of tool rounds: decline the pending calls and force an answer from what we have.
            messages += _tool_exchange(turn, [TOOL_BUDGET_EXHAUSTED] * len(turn.tool_calls))
            turn = await counted_chat(messages, tools=self._tools, tool_choice="none")
        if not (turn.content or "").strip():
            # Models occasionally end with an empty message; asking again usually produces the answer.
            turn = await counted_chat(messages, tools=self._tools, tool_choice="none")
        if state.sources and (uncited := uncited_sentences(_clean_answer(turn.content or ""), state.sources)):
            turn = await self._repair_citations(counted_chat, messages, turn, uncited, state.sources)
        return turn, state

    async def _repair_citations(self, chat: Chat, messages: list[dict[str, Any]], turn: AssistantTurn,
                                uncited: list[str], issued: Collection[str]) -> AssistantTurn:
        """Give the model one turn to cite or drop factual sentences that carry no [S#] id.

        The repair is optional polish: if the call fails, or its answer isn't better cited, the original
        answer is kept, so it never costs a good answer or a provider restart.

        Returns:
            The repaired turn if it leaves fewer uncited sentences, otherwise the original.
        """
        logger.info("Answer has %d uncited sentence(s), asking for a citation repair", len(uncited))
        listed = "\n".join(f"- {sentence}" for sentence in uncited)
        repair_messages = [*messages, {"role": "assistant", "content": turn.content},
                           {"role": "user", "content": CITATION_REPAIR.format(sentences=listed)}]
        try:
            repaired = await chat(repair_messages, tools=self._tools, tool_choice="none")
        except (LLMError, *FAILOVER_ERRORS) as error:
            logger.warning("Citation repair skipped: %s", type(error).__name__)
            return turn
        still_uncited = uncited_sentences(_clean_answer(repaired.content or ""), issued)
        if not (repaired.content or "").strip() or len(still_uncited) >= len(uncited):
            return turn
        return repaired

    async def _run_tool(self, call: ToolCall, state: _RequestState) -> str:
        """Execute one tool call; bad arguments and unknown tools become error text for the model."""
        logger.info("Tool call %s(%s)", call.name, call.arguments)
        if not state.is_first_call(call):
            return DUPLICATE_CALL
        try:
            args = json.loads(call.arguments)
            if call.name == ToolName.SEARCH_DOCUMENTS:
                return await self._search_documents(SearchArgs.model_validate(args), state)
            if call.name == ToolName.GET_SUPERHERO:
                return await self._get_superhero(HeroArgs.model_validate(args), state)
        except (json.JSONDecodeError, ValidationError) as error:
            return f"ERROR: invalid arguments for {call.name}: {error}"
        return f"ERROR: unknown tool {call.name}"

    async def _search_documents(self, args: SearchArgs, state: _RequestState) -> str:
        """Retrieve passages as source-labelled blocks, plus a hint if a searched name is also a comic character."""
        if args.collection is not None and args.collection not in self._collections:
            return f"ERROR: unknown collection '{args.collection}'. Available: {', '.join(sorted(self._collections))}."
        chunks = self._retriever.search(args.query, self._top_k, args.collection)
        blocks = []
        for chunk in chunks:
            title = f"{chunk.title} — {chunk.section}"
            source_id = state.add_source(SourceType.DOCUMENT, title, chunk.url, chunk.collection)
            blocks.append(f"[{source_id}] ({chunk.collection}) {title}\n{chunk.text}")
        result = "\n\n".join(blocks) or NO_PASSAGES  # the model must see an empty search as empty, hint or not
        if hint := await self._superhero_hint(args.query, state):
            result += f"\n\n{hint}"
        return result

    async def _superhero_hint(self, query: str, state: _RequestState) -> str | None:
        """Note which article names in the query are also comic-book characters, unless heroes are already being looked up."""
        if ToolName.GET_SUPERHERO in state.tools_used:
            return None
        candidates = self._titles.names_in(query, figures_only=True)[:MAX_HINT_LOOKUPS]
        lookups = await asyncio.gather(*(self._superheroes.search(name) for name in candidates), return_exceptions=True)
        names = []
        for name, heroes in zip(candidates, lookups, strict=True):
            if isinstance(heroes, SuperheroAPIError):
                # A hint is optional, so an outage here is logged rather than reported to the model.
                logger.warning("Superhero hint lookup for %r failed: %s", name, heroes)
                continue
            if isinstance(heroes, BaseException):
                raise heroes
            if any(hero.name.lower() == name.lower() for hero in heroes):
                names.append(f"'{name}'")
        return SUPERHERO_HINT.format(names=" and ".join(names), tool=ToolName.GET_SUPERHERO) if names else None

    async def _get_superhero(self, args: HeroArgs, state: _RequestState) -> str:
        """Look up a hero and format matches as source-labelled blocks; an API outage becomes a warning."""
        try:
            heroes = await self._superheroes.search(args.name)
        except SuperheroAPIError as error:
            state.warnings.append(str(error))
            return SUPERHERO_UNAVAILABLE
        hint = self._document_hint(args.name, state)
        if not heroes:
            return " ".join(filter(None, [f"No superhero named '{args.name}' was found.", hint]))
        blocks = []
        for hero in heroes:
            # No URL: the API's only per-hero URL embeds the private token. The id in the title is enough to look it up.
            source_id = state.add_source(SourceType.SUPERHERO_API, f"{hero.name} (Superhero API #{hero.id})", None)
            blocks.append(f"[{source_id}] {json.dumps(hero.profile, ensure_ascii=False, separators=(',', ':'))}")
        return "\n\n".join([*blocks, hint] if hint else blocks)

    def _document_hint(self, name: str, state: _RequestState) -> str | None:
        """Note the document collections with an article on this name, unless documents are already being searched."""
        if ToolName.SEARCH_DOCUMENTS in state.tools_used or not (collections := self._titles.collections_for(name)):
            return None
        return DOCUMENT_HINT.format(name=name, collections="', '".join(collections), tool=ToolName.SEARCH_DOCUMENTS)


def _attribute(answer: str, state: _RequestState, model: str) -> list[Source]:
    """Pick the sources to return: cited ones, else everything retrieved, else the model itself."""
    if cited := state.cited_sources(answer):
        return cited
    if state.sources:
        # Tools ran but the model forgot to cite: list everything it was given.
        return list(state.sources.values())
    # No tool was used, so the answer rests on the model alone. Say so rather than return nothing.
    return [Source(id="S0", type=SourceType.LLM, title=f"{model} (no supporting documents)")]


def uncited_sentences(answer: str, issued: Collection[str]) -> list[str]:
    """Factual sentences on lines (paragraphs or list items) that cite none of the issued [S#] ids.

    Citations are checked per line, not per sentence: one citation closing a paragraph is a normal way to
    source it. Only ids in issued count, so an invented [S9] doesn't. Skipped as non-factual: headings,
    lead-in lines ending in ":", table rows, fragments under MIN_FACT_WORDS words, and sentences saying the
    sources don't cover something (nothing to cite).
    """
    uncited = []
    for line in answer.replace("’", "'").splitlines():
        if line.lstrip().startswith("#"):
            continue
        text = _LIST_PREFIX_RE.sub("", line).replace("*", "").strip()
        cited = any(source_id in issued for source_id in _CITATION_RE.findall(text))
        if not text or text.endswith(":") or text.startswith("|") or cited:
            continue
        uncited += [sentence for sentence in _SENTENCE_SPLIT_RE.split(text)
                    if len(sentence.split()) >= MIN_FACT_WORDS and not _NO_ANSWER_RE.search(sentence)]
    return uncited


def _clean_answer(answer: str) -> str:
    """Strip invisible characters and rewrite citation markers into the [S1] form the response promises."""
    return _LOOSE_CITATION_RE.sub(r"[\1]", answer.translate(_CHARACTER_FIXES)).strip()


def _tool_exchange(turn: AssistantTurn, results: list[str]) -> list[dict[str, Any]]:
    """The assistant's tool-call message followed by one tool message per result, in OpenAI format."""
    assistant_message = {
        "role": "assistant",
        "content": turn.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
                **call.extra,
            }
            for call in turn.tool_calls
        ],
    }
    tool_messages = [
        {"role": "tool", "tool_call_id": call.id, "content": result}
        for call, result in zip(turn.tool_calls, results, strict=True)
    ]
    return [assistant_message, *tool_messages]
