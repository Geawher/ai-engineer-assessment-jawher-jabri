from __future__ import annotations

import asyncio
import json
import logging
import re
from enum import StrEnum
from typing import Annotated, Any, Protocol

from pydantic import BaseModel, StringConstraints, ValidationError

from app.llm import AssistantTurn, Chat, LLMClient, TokenUsage, ToolCall
from app.retrieval.base import Retriever
from app.schemas import AskResponse, Source, SourceType
from app.sources.superhero import Hero, SuperheroAPIError

logger = logging.getLogger(__name__)


class ToolName(StrEnum):
    SEARCH_MYTHOLOGY = "search_mythology"
    GET_SUPERHERO = "get_superhero"


SYSTEM_PROMPT = f"""\
You answer questions about Norse and Greek mythology and comic-book superheroes, using only your tools.

Rules:
1. Use {ToolName.SEARCH_MYTHOLOGY} for myths, gods and legendary heroes, {ToolName.GET_SUPERHERO} for comic-book \
characters. Many names exist in both (Thor, Loki, Odin, Hercules, Medusa...): unless the question limits itself \
to one, use both and cover both.
2. Search once per figure; one search rarely covers two.
3. Answer only from tool results. If they don't contain the answer, say you don't know; if they contradict \
the question's premise, say so.
4. Cite the [S#] id of each fact right after it, e.g. "Thor wields Mjölnir [S2]."
5. For questions outside mythology and superheroes, call no tools and say you only cover those topics.
6. Your final message must hold the whole answer: text sent alongside tool calls is not shown.
7. Be concise."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": ToolName.SEARCH_MYTHOLOGY,
            "description": "Keyword search over Wikipedia articles on Norse and Greek mythology. "
                           "Name the figure explicitly; one call per figure.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Keywords, e.g. 'Thor hammer Mjolnir'."}},
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
and remove any statement no tool result supports. Keep everything else unchanged. \
Reply with the full corrected answer only."""
DUPLICATE_CALL = "You already made this exact call; its result is above. Use it, or try a different query or tool."
SUPERHERO_UNAVAILABLE = "ERROR: the Superhero API is unavailable right now. Tell the user superhero data could not be retrieved."
_CITATION_RE = re.compile(r"\bS\d+\b")
# Models drift from the requested [S1] form, e.g. gpt-oss writes 【S1】, 【S5†L9-L12】 or [ S1 ].
_LOOSE_CITATION_RE = re.compile(r"[【\[]\s*(S\d+)(?:†[^】\]]*)?\s*[】\]]")
# Invisible or typographic characters some models emit; they break copy-paste and string matching.
# Splitting an answer into checkable sentences: markdown list/heading markers, then sentence ends.
_MARKDOWN_PREFIX_RE = re.compile(r"^\s*(?:#+|[-*•]|\d+[.)])\s*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z*])")
# Sentences saying the sources don't cover something have nothing to cite.
_NO_ANSWER_RE = re.compile(r"don't know|do not know|not (?:found|mentioned|available|specified|provided|listed)|"
                           r"no (?:information|record|data|mention|details)|(?:don't|do not|doesn't|does not) "
                           r"(?:contain|mention|say|give|provide|include)|can(?:'|no)t|unable|only (?:answer|cover)|sorry|"
                           r"there is no|\bno\b.*\b(?:was|were) found",
                           re.IGNORECASE)
MIN_FACT_WORDS = 6  # shorter fragments ("In mythology:", "Yes.") aren't claims worth citing
_CHARACTER_FIXES = str.maketrans({"‐": "-", "‑": "-", "​": None, "‌": None, "‍": None})


class SearchArgs(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


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

    def add_source(self, source_type: SourceType, title: str, url: str | None) -> str:
        """Register a source and return the id ("S1", "S2", ...) the model should cite it by.

        Passages from the same section share one id, so the response never lists the same source twice.
        """
        for source in self.sources.values():
            if (source.type, source.title, source.url) == (source_type, title, url):
                return source.id
        source_id = f"S{len(self.sources) + 1}"
        self.sources[source_id] = Source(id=source_id, type=source_type, title=title, url=url)
        return source_id

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
                 top_k: int = 4, max_tool_rounds: int = 3):
        """Create an agent.

        Args:
            llm: Chat client used for routing and answering.
            retriever: Mythology passage retriever.
            superheroes: Superhero lookup source.
            top_k: Passages returned per mythology search.
            max_tool_rounds: Model turns allowed to call tools before an answer is forced.
        """
        self._llm = llm
        self._retriever = retriever
        self._superheroes = superheroes
        self._top_k = top_k
        self._max_tool_rounds = max_tool_rounds

    async def ask(self, question: str) -> AskResponse:
        """Answer a question with sources.

        Raises:
            LLMError: If the LLM can't be reached or rejects the request; tool failures don't raise.
        """
        turn, state = await self._llm.run_conversation(lambda chat: self._converse(chat, question))
        logger.info("Answered with %d LLM calls: %d prompt tokens (%d cached), %d completion tokens",
                    state.llm_calls, state.usage.prompt, state.usage.cached_prompt, state.usage.completion)
        answer = _clean_answer(turn.content or "") or NO_ANSWER
        if state.sources and (still_uncited := uncited_sentences(answer)):
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        turn = await counted_chat(messages, tools=TOOLS)
        rounds = 0
        while turn.tool_calls and rounds < self._max_tool_rounds:
            results = await asyncio.gather(*(self._run_tool(call, state) for call in turn.tool_calls))
            messages += _tool_exchange(turn, results)
            turn = await counted_chat(messages, tools=TOOLS)
            rounds += 1

        if turn.tool_calls:
            # Out of tool rounds: decline the pending calls and force an answer from what we have.
            messages += _tool_exchange(turn, [TOOL_BUDGET_EXHAUSTED] * len(turn.tool_calls))
            turn = await counted_chat(messages, tools=TOOLS, tool_choice="none")
        if not (turn.content or "").strip():
            # Models occasionally end with an empty message; asking again usually produces the answer.
            turn = await counted_chat(messages, tools=TOOLS, tool_choice="none")
        if state.sources and (uncited := uncited_sentences(_clean_answer(turn.content or ""))):
            turn = await self._repair_citations(counted_chat, messages, turn, uncited)
        return turn, state

    async def _repair_citations(self, chat: Chat, messages: list[dict[str, Any]], turn: AssistantTurn,
                                uncited: list[str]) -> AssistantTurn:
        """Give the model one turn to cite or drop factual sentences that carry no [S#] id.

        Returns:
            The corrected turn, or the original if the model returned nothing usable.
        """
        logger.info("Answer has %d uncited sentence(s), asking for a citation repair", len(uncited))
        listed = "\n".join(f"- {sentence}" for sentence in uncited)
        repair_messages = [*messages, {"role": "assistant", "content": turn.content},
                           {"role": "user", "content": CITATION_REPAIR.format(sentences=listed)}]
        repaired = await chat(repair_messages, tools=TOOLS, tool_choice="none")
        return repaired if (repaired.content or "").strip() else turn

    async def _run_tool(self, call: ToolCall, state: _RequestState) -> str:
        """Execute one tool call; bad arguments and unknown tools become error text for the model."""
        if call.name not in state.tools_used:
            state.tools_used.append(call.name)
        logger.info("Tool call %s(%s)", call.name, call.arguments)
        if not state.is_first_call(call):
            return DUPLICATE_CALL
        try:
            args = json.loads(call.arguments)
            if call.name == ToolName.SEARCH_MYTHOLOGY:
                return self._search_mythology(SearchArgs.model_validate(args), state)
            if call.name == ToolName.GET_SUPERHERO:
                return await self._get_superhero(HeroArgs.model_validate(args), state)
        except (json.JSONDecodeError, ValidationError) as error:
            return f"ERROR: invalid arguments for {call.name}: {error}"
        return f"ERROR: unknown tool {call.name}"

    def _search_mythology(self, args: SearchArgs, state: _RequestState) -> str:
        """Retrieve passages and format them as source-labelled blocks."""
        chunks = self._retriever.search(args.query, self._top_k)
        if not chunks:
            return "No matching mythology passages found."
        blocks = []
        for chunk in chunks:
            title = f"{chunk.title} — {chunk.section}"
            source_id = state.add_source(SourceType.MYTHOLOGY_DATASET, title, chunk.url)
            blocks.append(f"[{source_id}] {title}\n{chunk.text}")
        return "\n\n".join(blocks)

    async def _get_superhero(self, args: HeroArgs, state: _RequestState) -> str:
        """Look up a hero and format matches as source-labelled blocks; an API outage becomes a warning."""
        try:
            heroes = await self._superheroes.search(args.name)
        except SuperheroAPIError as error:
            state.warnings.append(str(error))
            return SUPERHERO_UNAVAILABLE
        if not heroes:
            return (f"No superhero named '{args.name}' was found. If the question is about mythology, "
                    f"use {ToolName.SEARCH_MYTHOLOGY} instead.")
        blocks = []
        for hero in heroes:
            # No URL: the API's only per-hero URL embeds the private token. The id in the title is enough to look it up.
            source_id = state.add_source(SourceType.SUPERHERO_API, f"{hero.name} (Superhero API #{hero.id})", None)
            blocks.append(f"[{source_id}] {json.dumps(hero.profile, ensure_ascii=False, separators=(',', ':'))}")
        return "\n\n".join(blocks)


def _attribute(answer: str, state: _RequestState, model: str) -> list[Source]:
    """Pick the sources to return: cited ones, else everything retrieved, else the model itself."""
    if cited := state.cited_sources(answer):
        return cited
    if state.sources:
        # Tools ran but the model forgot to cite: list everything it was given.
        return list(state.sources.values())
    # No tool was used, so the answer rests on the model alone. Say so rather than return nothing.
    return [Source(id="S0", type=SourceType.LLM, title=f"{model} (no supporting documents)")]


def uncited_sentences(answer: str) -> list[str]:
    """Factual sentences in paragraphs (or list items) that carry no [S#] citation at all.

    Citations are checked per paragraph, not per sentence: one citation closing a paragraph is a normal
    way to source it. Skipped as non-factual: headings, lead-in lines ending in ":", fragments under
    MIN_FACT_WORDS words, and sentences saying the sources don't cover something (nothing to cite).
    """
    uncited = []
    for line in answer.splitlines():
        paragraph = _MARKDOWN_PREFIX_RE.sub("", line).replace("*", "").strip()
        if not paragraph or paragraph.endswith(":") or paragraph.startswith("|") or _CITATION_RE.search(paragraph):
            continue
        uncited += [sentence for sentence in _SENTENCE_SPLIT_RE.split(paragraph)
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
