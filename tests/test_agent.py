from __future__ import annotations

import json

import httpx
import openai
import pytest

from app.agent import ALREADY_SHOWN, DUPLICATE_CALL, Agent, ToolName, _excerpt, _trim_profile, uncited_sentences
from app.llm import AssistantTurn, ToolCall
from app.retrieval.base import Chunk
from app.retrieval.bm25 import BM25Retriever
from app.schemas import SourceType
from app.sources.superhero import Hero
from app.retrieval.titles import TitleIndex
from tests.conftest import CHUNKS, COLLECTIONS, THOR_HERO, FakeSuperheroes, ScriptedLLM, answer_turn, tool_turn


def make_agent(llm: ScriptedLLM, retriever: BM25Retriever, superheroes: FakeSuperheroes | None = None,
               **options: int) -> Agent:
    """Build an agent over fakes; options are forwarded as keyword arguments.

    Pre-fetching is off unless a test turns it on, so the scripted turns describe the whole tool loop.
    """
    return Agent(llm, retriever, superheroes or FakeSuperheroes(), collections=COLLECTIONS,
                 titles=TitleIndex(CHUNKS), **({"prefetch": False} | options))


async def test_mythology_question_cites_dataset(retriever):
    """A mythology question returns the cited dataset passage, labelled with the id the model saw."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Thor hammer"})),
        answer_turn("Thor wields Mjölnir [S1]."),
    ])
    response = await make_agent(llm, retriever).ask("What weapon does Thor use?")

    assert response.tools_used == [ToolName.SEARCH_DOCUMENTS]
    assert [source.type for source in response.sources] == [SourceType.DOCUMENT]
    assert response.sources[0].title == "Thor — Introduction"
    tool_message = llm.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["content"].startswith("[S1] (mythology) Thor")


async def test_ambiguous_question_uses_both_sources(retriever):
    """When the model calls both tools, sources from both are returned."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Thor"}), (ToolName.GET_SUPERHERO, {"name": "Thor"})),
        answer_turn("In myth Thor is a thunder god [S1]; in comics he is published by Marvel [S2]."),
    ])
    agent = make_agent(llm, retriever, FakeSuperheroes({"thor": [THOR_HERO]}))
    response = await agent.ask("Who is Thor?")

    assert set(response.tools_used) == {ToolName.SEARCH_DOCUMENTS, ToolName.GET_SUPERHERO}
    assert {source.type for source in response.sources} == {SourceType.DOCUMENT, SourceType.SUPERHERO_API}


async def test_only_cited_sources_are_returned(retriever):
    """Retrieved-but-uncited passages are left out of the response."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "thunder god"})),  # matches Thor and Zeus
        answer_turn("Zeus is king of the gods [S2]."),
    ])
    response = await make_agent(llm, retriever).ask("Who is king of the Greek gods?")
    assert [source.id for source in response.sources] == ["S2"]


async def test_invented_citation_is_ignored(retriever):
    """A citation id the server never issued is dropped."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Odin"})),
        answer_turn("Odin is wise [S1] and also [S9]."),
    ])
    response = await make_agent(llm, retriever).ask("Who is Odin?")
    assert [source.id for source in response.sources] == ["S1"]


async def test_uncited_answer_falls_back_to_all_retrieved_sources(retriever):
    """If the model forgets to cite, every retrieved source is listed."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Odin"})),
        answer_turn("Odin is associated with wisdom."),
    ])
    response = await make_agent(llm, retriever).ask("Who is Odin?")
    assert response.sources
    assert all(source.type == SourceType.DOCUMENT for source in response.sources)


async def test_no_tools_marks_answer_as_llm_only(retriever):
    """An answer given without tools is attributed to the model itself."""
    llm = ScriptedLLM([answer_turn("I can only answer questions about mythology and superheroes.")])
    response = await make_agent(llm, retriever).ask("What's the weather in Paris?")

    assert response.tools_used == []
    assert [source.type for source in response.sources] == [SourceType.LLM]


async def test_superhero_api_failure_degrades_gracefully(retriever):
    """A Superhero API outage becomes a warning and an error message to the model, not a failed request."""
    llm = ScriptedLLM([
        tool_turn((ToolName.GET_SUPERHERO, {"name": "Batman"})),
        answer_turn("Sorry, superhero data is unavailable right now."),
    ])
    response = await make_agent(llm, retriever, FakeSuperheroes(is_down=True)).ask("How strong is Batman?")

    assert response.warnings == ["Superhero API is unavailable"]
    assert "ERROR" in llm.calls[1]["messages"][-1]["content"]


async def test_invalid_tool_arguments_are_reported_to_the_model(retriever):
    """Malformed tool arguments are returned to the model as an error instead of raising."""
    bad_call = AssistantTurn(content=None, model="m",
                             tool_calls=[ToolCall(id="c1", name=ToolName.GET_SUPERHERO, arguments="{not json")])
    llm = ScriptedLLM([bad_call, answer_turn("I couldn't look that up.")])
    await make_agent(llm, retriever).ask("Who is Batman?")
    assert llm.calls[1]["messages"][-1]["content"].startswith("ERROR: invalid arguments")


async def test_tool_loop_is_bounded(retriever):
    """After max_tool_rounds the model is forced to answer without tools."""
    looping = tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Thor"}))
    llm = ScriptedLLM([looping, looping, answer_turn("Thor is a god [S1].")])
    response = await make_agent(llm, retriever, max_tool_rounds=1).ask("Who is Thor?")

    assert llm.calls[-1]["tool_choice"] == "none"
    assert response.answer == "Thor is a god [S1]."


async def test_passages_from_same_section_share_one_source(retriever):
    """Two searches hitting the same section produce one source id, not duplicates."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Thor"}), (ToolName.SEARCH_DOCUMENTS, {"query": "Mjolnir"})),
        answer_turn("Thor wields Mjölnir [S1]."),
    ])
    response = await make_agent(llm, retriever).ask("What does Thor wield?")
    assert [source.id for source in response.sources] == ["S1"]


async def test_lenticular_citations_are_normalized(retriever):
    """Drifting citation formats and invisible characters are normalized; ids still count."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Odin"})),
        answer_turn("Odin is wise【S1】, old [ S1 ] and one\u2011eyed【S1†L9-L12】 [\u200bS1]."),
    ])
    response = await make_agent(llm, retriever).ask("Who is Odin?")
    assert response.answer == "Odin is wise[S1], old [S1] and one-eyed[S1] [S1]."
    assert [source.id for source in response.sources] == ["S1"]


async def test_provider_specific_tool_call_fields_are_echoed_back(retriever):
    """Opaque tool-call fields (e.g. Gemini thought signatures) are sent back in the next request."""
    signature = {"extra_content": {"google": {"thought_signature": "sig"}}}
    call = ToolCall(id="c1", name=ToolName.SEARCH_DOCUMENTS, arguments='{"query": "Odin"}', extra=signature)
    llm = ScriptedLLM([AssistantTurn(content=None, model="m", tool_calls=[call]), answer_turn("Odin [S1].")])
    await make_agent(llm, retriever).ask("Who is Odin?")

    assistant_message = llm.calls[1]["messages"][-2]
    assert assistant_message["tool_calls"][0]["extra_content"] == signature["extra_content"]


async def test_repeated_identical_tool_call_is_short_circuited(retriever):
    """The same call twice in one request is not re-executed; the model is told to use the first result."""
    search = (ToolName.SEARCH_DOCUMENTS, {"query": "Odin"})
    llm = ScriptedLLM([tool_turn(search), tool_turn(search), answer_turn("Odin is wise [S1].")])
    await make_agent(llm, retriever).ask("Who is Odin?")
    assert llm.calls[2]["messages"][-1]["content"] == DUPLICATE_CALL


async def test_empty_final_answer_is_retried_once(retriever):
    """An empty final message triggers one more call with tools disabled."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Odin"})),
        answer_turn(""),
        answer_turn("Odin is wise [S1]."),
    ])
    response = await make_agent(llm, retriever).ask("Who is Odin?")
    assert response.answer == "Odin is wise [S1]."
    assert llm.calls[-1]["tool_choice"] == "none"


async def test_missing_hero_with_an_article_hints_at_the_documents(retriever):
    """A hero lookup for a name that has an article (Zeus) points the model to the document collection."""
    llm = ScriptedLLM([tool_turn((ToolName.GET_SUPERHERO, {"name": "Zeus"})), answer_turn("...")])
    await make_agent(llm, retriever).ask("Who is Zeus?")
    result = llm.calls[1]["messages"][-1]["content"]
    assert "No superhero named 'Zeus'" in result and "'mythology'" in result and ToolName.SEARCH_DOCUMENTS in result


async def test_no_hint_for_names_no_other_source_knows(retriever):
    """A missing hero with no article gets a plain "not found", not a speculative hint."""
    llm = ScriptedLLM([tool_turn((ToolName.GET_SUPERHERO, {"name": "Achilles"})), answer_turn("...")])
    await make_agent(llm, retriever).ask("What was Achilles' weak spot?")
    assert llm.calls[1]["messages"][-1]["content"] == "No superhero named 'Achilles' was found."


async def test_search_hints_when_a_name_is_also_a_comic_character(retriever):
    """Searching documents for Thor notes that Thor is also in the superhero database."""
    llm = ScriptedLLM([tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Thor hammer"})), answer_turn("...")])
    await make_agent(llm, retriever, FakeSuperheroes({"thor": [THOR_HERO]})).ask("Who is Thor?")
    result = llm.calls[1]["messages"][-1]["content"]
    assert "'Thor' also exist(s) in the superhero database" in result


async def test_no_hint_for_a_source_already_being_queried(retriever):
    """Both tools in the same turn: neither result carries a redundant hint."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Thor"}), (ToolName.GET_SUPERHERO, {"name": "Thor"})),
        answer_turn("..."),
    ])
    await make_agent(llm, retriever, FakeSuperheroes({"thor": [THOR_HERO]})).ask("Who is Thor?")
    assert all("Note:" not in message["content"] for message in llm.calls[1]["messages"] if message["role"] == "tool")


async def test_search_can_target_one_collection_and_rejects_unknown_ones(retriever):
    """A known collection filters results and is recorded on the source; an unknown one is an error for the model."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Odin", "collection": "mythology"})),
        tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Odin", "collection": "recipes"})),
        answer_turn("Odin is wise [S1]."),
    ])
    response = await make_agent(llm, retriever).ask("Who is Odin?")
    assert response.sources[0].collection == "mythology"
    assert llm.calls[2]["messages"][-1]["content"].startswith("ERROR: unknown collection 'recipes'")


class TestCitationRepair:
    def test_paragraph_with_a_citation_counts_as_cited(self):
        """One citation closing a paragraph sources the whole paragraph."""
        answer = "Thor is the Norse god of thunder. He rides a chariot pulled by two goats [S1]."
        assert uncited_sentences(answer, {"S1"}) == []

    def test_uncited_factual_paragraph_is_flagged(self):
        """A paragraph of claims with no [S#] anywhere is flagged sentence by sentence."""
        answer = "Odin is the father of Thor [S1].\n\nBoth figures are associated with storms and with kingship in later stories."
        assert uncited_sentences(answer, {"S1"}) == ["Both figures are associated with storms and with kingship in later stories."]

    @pytest.mark.parametrize("answer", [
        "## In the comics:",
        "Yes, he does.",
        "The sources don't mention Zeus's favourite food at all.",
        "No comic-book character named \"Apollo\" was found in the database.",
        "I can only answer questions about mythology and superheroes.",
    ])
    def test_non_factual_lines_are_not_flagged(self, answer):
        """Headings, short fragments and "not in the sources" statements have nothing to cite."""
        assert uncited_sentences(answer, {"S1"}) == []

    async def test_uncited_answer_gets_one_repair_turn(self, retriever):
        """An uncited claim triggers one repair call, whose answer replaces the original."""
        llm = ScriptedLLM([
            tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Odin"})),
            answer_turn("Odin is a widely revered god associated with wisdom and healing."),
            answer_turn("Odin is a widely revered god associated with wisdom and healing [S1]."),
        ])
        response = await make_agent(llm, retriever).ask("Who is Odin?")

        assert response.answer.endswith("[S1].")
        assert "no [S#] citation" in llm.calls[-1]["messages"][-1]["content"]
        assert llm.calls[-1]["tool_choice"] == "none"
        assert response.warnings == []

    async def test_still_uncited_after_repair_is_flagged(self, retriever):
        """If the repair doesn't add citations, the response says so in warnings."""
        claim = "Odin is a widely revered god associated with wisdom and healing."
        llm = ScriptedLLM([tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Odin"})), answer_turn(claim), answer_turn(claim)])
        response = await make_agent(llm, retriever).ask("Who is Odin?")
        assert response.warnings == ["1 statement(s) in the answer are not linked to a source."]

    async def test_no_repair_without_sources(self, retriever):
        """An uncited factual answer given without tools never triggers a repair (there is nothing to cite)."""
        llm = ScriptedLLM([answer_turn("Paris is the capital and largest city of France by population.")])
        await make_agent(llm, retriever).ask("What is the capital of France?")
        assert len(llm.calls) == 1

    async def test_repair_call_failure_keeps_the_original_answer(self, retriever):
        """A failing repair call (rate limit here) neither fails the request nor restarts it."""
        claim = "Odin is a widely revered god associated with wisdom and healing."
        llm = FailingRepairLLM([tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Odin"})), answer_turn(claim)])
        response = await make_agent(llm, retriever).ask("Who is Odin?")
        assert response.answer == claim
        assert response.warnings == ["1 statement(s) in the answer are not linked to a source."]

    async def test_repair_that_is_not_better_is_discarded(self, retriever):
        """A repaired answer with as many uncited sentences as before doesn't replace the original."""
        claim = "Odin is a widely revered god associated with wisdom and healing."
        worse = "Odin is a god of many things in the old northern stories of the region."
        llm = ScriptedLLM([tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Odin"})), answer_turn(claim), answer_turn(worse)])
        response = await make_agent(llm, retriever).ask("Who is Odin?")
        assert response.answer == claim

    async def test_repair_turn_message_order(self, retriever):
        """Exactly one repair call, appended as assistant answer then user instruction after the tool messages."""
        claim = "Odin is a widely revered god associated with wisdom and healing."
        llm = ScriptedLLM([tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Odin"})), answer_turn(claim),
                           answer_turn(claim[:-1] + " [S1].")])
        await make_agent(llm, retriever).ask("Who is Odin?")
        assert len(llm.calls) == 3
        roles = [message["role"] for message in llm.calls[-1]["messages"]]
        assert roles[-4:] == ["assistant", "tool", "assistant", "user"]

    @pytest.mark.parametrize(("answer", "flagged"), [
        ("I don’t know who that is, the sources say nothing.", False),                 # curly apostrophe
        ("Superhero data could not be retrieved right now, sorry about that.", False),  # outage notice
        ("## Thor in Norse mythology and in Marvel comics", False),                     # long heading
        ("Achilles cannot be harmed anywhere except at his heel.", True),               # claim, not a refusal
        ("There is no god more feared than Fenrir in the whole of Asgard.", True),
        ("Odin is wise and rules over Asgard from his high seat [S9].", True),          # S9 was never issued
    ])
    def test_detector_edge_cases_from_review(self, answer, flagged):
        """Refusal-like wording is only skipped when it is about the sources or the speaker."""
        assert bool(uncited_sentences(answer, {"S1"})) is flagged

    def test_sentences_split_before_non_ascii_capitals(self):
        """A sentence starting with "Æsir" is split from the one before it."""
        answer = "Odin rules the gods of the north from Asgard. Æsir and Vanir once fought a long war together."
        assert len(uncited_sentences(answer, set())) == 2



async def test_hint_uses_the_alias_that_matched(retriever):
    """Searching "Ajax" (article "Ajax the Great") checks the superhero database for "Ajax", not the full title."""
    chunks = [Chunk("ajax:0", "mythology", "Ajax the Great", "Introduction", "https://x/Ajax", "Ajax fought at Troy.")]
    ajax = Hero(id="13", name="Ajax", profile={"name": "Ajax"})
    llm = ScriptedLLM([tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Ajax Trojan war"})), answer_turn("...")])
    agent = Agent(llm, BM25Retriever(chunks), FakeSuperheroes({"ajax": [ajax]}), collections=COLLECTIONS,
                  titles=TitleIndex(chunks), prefetch=False)
    await agent.ask("Who is Ajax?")
    assert "'Ajax' also exist(s) in the superhero database" in llm.calls[1]["messages"][-1]["content"]


class EmptyRetriever:
    """Retriever that never finds anything (e.g. a collection filter that excludes every match)."""

    def search(self, query: str, k: int, collection: str | None = None) -> list[Chunk]:
        """Return no passages."""
        return []


async def test_empty_search_still_says_so_when_a_hint_is_added(retriever):
    """No passages plus a hint: the model is told the search was empty, then gets the hint."""
    llm = ScriptedLLM([tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Thor xyzzy"})), answer_turn("...")])
    agent = Agent(llm, EmptyRetriever(), FakeSuperheroes({"thor": [THOR_HERO]}), collections=COLLECTIONS,
                  titles=TitleIndex(CHUNKS), prefetch=False)
    await agent.ask("Thor xyzzy?")
    result = llm.calls[1]["messages"][-1]["content"]
    assert result.startswith("No matching passages found.") and "Note:" in result


async def test_unknown_tools_are_not_reported_as_used(retriever):
    """A model calling a tool that doesn't exist gets an error, and the tool isn't listed in tools_used."""
    llm = ScriptedLLM([tool_turn(("search_mythology", {"query": "Thor"})), answer_turn("I could not search.")])
    response = await make_agent(llm, retriever).ask("Who is Thor?")
    assert response.tools_used == []
    assert llm.calls[1]["messages"][-1]["content"] == "ERROR: unknown tool search_mythology"


class FailingRepairLLM(ScriptedLLM):
    """Scripted LLM whose call after the scripted turns raises a rate limit, like a provider during the repair turn."""

    async def complete(self, messages, tools=None, tool_choice="auto"):
        """Return scripted turns, then fail."""
        if not self._turns:
            raise openai.RateLimitError("slow down", response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
                                        body=None)
        return await super().complete(messages, tools=tools, tool_choice=tool_choice)


class TestPrefetch:
    async def test_names_in_the_question_are_fetched_before_the_first_call(self, retriever):
        """'Who is Thor?' arrives with the Thor passages and the Thor hero already in the conversation."""
        llm = ScriptedLLM([answer_turn("Thor is a thunder god [S1], published by Marvel [S2].")])
        agent = make_agent(llm, retriever, FakeSuperheroes({"thor": [THOR_HERO]}), prefetch=True)
        response = await agent.ask("Who is Thor?")

        assert len(llm.calls) == 1  # one LLM call instead of a tool round plus an answer
        messages = llm.calls[0]["messages"]
        calls = messages[2]["tool_calls"]
        assert [(call["function"]["name"], json.loads(call["function"]["arguments"])) for call in calls] == [
            (ToolName.SEARCH_DOCUMENTS, {"query": "Who is Thor?"}), (ToolName.GET_SUPERHERO, {"name": "Thor"})]
        assert messages[3]["content"].startswith("[S1] (mythology) Thor") and "Note:" not in messages[3]["content"]
        assert {source.type for source in response.sources} == {SourceType.DOCUMENT, SourceType.SUPERHERO_API}

    async def test_each_figure_gets_its_own_search(self, retriever):
        """Two figures, two searches, each led by its figure's name."""
        llm = ScriptedLLM([answer_turn("...")])
        await make_agent(llm, retriever, prefetch=True).ask("Is Odin the father of Thor?")
        queries = [json.loads(call["function"]["arguments"])["query"] for call in llm.calls[0]["messages"][2]["tool_calls"]]
        assert queries == ["Odin Is Odin the father of Thor?", "Thor Is Odin the father of Thor?"]

    async def test_only_exact_superhero_names_are_fetched(self, retriever):
        """A capitalized word that only partly matches a hero (or is a question word) adds no lookup."""
        heroes = FakeSuperheroes({"batman": [Hero(id="69", name="Batman", profile={})],
                                  "compare": [Hero(id="1", name="Compare Man", profile={})]})
        llm = ScriptedLLM([answer_turn("...")])
        await make_agent(llm, retriever, heroes, prefetch=True).ask("Compare Batman's intelligence. What now?")
        calls = llm.calls[0]["messages"][2]["tool_calls"]
        assert [json.loads(call["function"]["arguments"]) for call in calls] == [{"name": "Batman"}]

    async def test_question_without_names_goes_straight_to_the_model(self, retriever):
        """No names, no tool exchange: off-topic questions cost exactly what they did before."""
        llm = ScriptedLLM([answer_turn("I only cover mythology and superheroes.")])
        await make_agent(llm, retriever, prefetch=True).ask("What is 17 multiplied by 23?")
        assert [message["role"] for message in llm.calls[0]["messages"]] == ["system", "user"]

    async def test_superhero_outage_during_prefetch_still_searches_documents(self, retriever):
        """The lookup is a shortcut: if the API is down, the documents are still fetched and nothing is raised."""
        llm = ScriptedLLM([answer_turn("Thor is a thunder god [S1].")])
        await make_agent(llm, retriever, FakeSuperheroes(is_down=True), prefetch=True).ask("Who is Thor?")
        calls = llm.calls[0]["messages"][2]["tool_calls"]
        assert [call["function"]["name"] for call in calls] == [ToolName.SEARCH_DOCUMENTS]

    async def test_repeating_a_prefetched_call_is_short_circuited(self, retriever):
        """The model asking again for what it was given gets told it already has it."""
        llm = ScriptedLLM([tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Who is Thor?"})), answer_turn("...")])
        await make_agent(llm, retriever, prefetch=True).ask("Who is Thor?")
        assert llm.calls[1]["messages"][-1]["content"] == DUPLICATE_CALL


class TestTrimming:
    RELATIVES = "Lois Lane (wife), Jor-El (father, deceased), Lara (mother), Jonathan Kent (adoptive father), Zor-El (uncle)"

    def test_long_list_fields_keep_their_first_entries(self):
        """Fields the question doesn't ask about keep MAX_LIST_ENTRIES entries; commas inside brackets don't split."""
        profile = {"relatives": self.RELATIVES, "aliases": ["a", "b", "c", "d", "e", "f"], "base": "Metropolis"}
        trimmed = _trim_profile(profile, "How tall is Superman?")
        assert trimmed["relatives"] == ("Lois Lane (wife), Jor-El (father, deceased), Lara (mother), "
                                        "Jonathan Kent (adoptive father), (+1 more)")
        assert trimmed["aliases"] == ["a", "b", "c", "d", "(+2 more)"]
        assert trimmed["base"] == "Metropolis" and profile["relatives"] == self.RELATIVES  # input left untouched

    @pytest.mark.parametrize("question", ["Who is Superman married to?", "Who is Superman's uncle?"])
    def test_fields_the_question_asks_about_stay_whole(self, question):
        """Family questions keep every relative."""
        assert _trim_profile({"relatives": self.RELATIVES}, question)["relatives"] == self.RELATIVES

    def test_excerpt_keeps_the_first_sentence_and_query_matches(self):
        """Sentences without a query word are cut and the gaps marked."""
        text = "Baldr is a god. He is loved by all. Hodr threw mistletoe at Baldr. The gods wept."
        assert _excerpt(text, "Baldr mistletoe") == "Baldr is a god. … Hodr threw mistletoe at Baldr. …"
        assert _excerpt(text, "loved gods") == "Baldr is a god. He is loved by all. … The gods wept."

    async def test_a_passage_is_sent_once_per_question(self, retriever):
        """A second search returning the same passage gets a pointer to it, under the same source id."""
        llm = ScriptedLLM([tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Thor"})),
                           tool_turn((ToolName.SEARCH_DOCUMENTS, {"query": "Thor Mjolnir"})), answer_turn("...")])
        await make_agent(llm, retriever).ask("What weapon does Thor use?")
        second = llm.calls[2]["messages"][-1]["content"]
        assert second.startswith(f"[S1] (mythology) Thor — Introduction\n{ALREADY_SHOWN}")
