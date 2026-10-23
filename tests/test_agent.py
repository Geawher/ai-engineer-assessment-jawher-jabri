from __future__ import annotations

from app.agent import DUPLICATE_CALL, Agent, ToolName
from app.llm import AssistantTurn, ToolCall
from app.retrieval.bm25 import BM25Retriever
from app.schemas import SourceType
from tests.conftest import THOR_HERO, FakeSuperheroes, ScriptedLLM, answer_turn, tool_turn


def make_agent(llm: ScriptedLLM, retriever: BM25Retriever, superheroes: FakeSuperheroes | None = None,
               **options: int) -> Agent:
    """Build an agent over fakes; options are forwarded as keyword arguments."""
    return Agent(llm, retriever, superheroes or FakeSuperheroes(), **options)


async def test_mythology_question_cites_dataset(retriever):
    """A mythology question returns the cited dataset passage, labelled with the id the model saw."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_MYTHOLOGY, {"query": "Thor hammer"})),
        answer_turn("Thor wields Mjölnir [S1]."),
    ])
    response = await make_agent(llm, retriever).ask("What weapon does Thor use?")

    assert response.tools_used == [ToolName.SEARCH_MYTHOLOGY]
    assert [source.type for source in response.sources] == [SourceType.MYTHOLOGY_DATASET]
    assert response.sources[0].title == "Thor — Introduction"
    tool_message = llm.calls[1]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["content"].startswith("[S1] Thor")


async def test_ambiguous_question_uses_both_sources(retriever):
    """When the model calls both tools, sources from both are returned."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_MYTHOLOGY, {"query": "Thor"}), (ToolName.GET_SUPERHERO, {"name": "Thor"})),
        answer_turn("In myth Thor is a thunder god [S1]; in comics he is published by Marvel [S2]."),
    ])
    agent = make_agent(llm, retriever, FakeSuperheroes({"thor": [THOR_HERO]}))
    response = await agent.ask("Who is Thor?")

    assert set(response.tools_used) == {ToolName.SEARCH_MYTHOLOGY, ToolName.GET_SUPERHERO}
    assert {source.type for source in response.sources} == {SourceType.MYTHOLOGY_DATASET, SourceType.SUPERHERO_API}


async def test_only_cited_sources_are_returned(retriever):
    """Retrieved-but-uncited passages are left out of the response."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_MYTHOLOGY, {"query": "thunder god"})),  # matches Thor and Zeus
        answer_turn("Zeus is king of the gods [S2]."),
    ])
    response = await make_agent(llm, retriever).ask("Who is king of the Greek gods?")
    assert [source.id for source in response.sources] == ["S2"]


async def test_invented_citation_is_ignored(retriever):
    """A citation id the server never issued is dropped."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_MYTHOLOGY, {"query": "Odin"})),
        answer_turn("Odin is wise [S1] and also [S9]."),
    ])
    response = await make_agent(llm, retriever).ask("Who is Odin?")
    assert [source.id for source in response.sources] == ["S1"]


async def test_uncited_answer_falls_back_to_all_retrieved_sources(retriever):
    """If the model forgets to cite, every retrieved source is listed."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_MYTHOLOGY, {"query": "Odin"})),
        answer_turn("Odin is associated with wisdom."),
    ])
    response = await make_agent(llm, retriever).ask("Who is Odin?")
    assert response.sources
    assert all(source.type == SourceType.MYTHOLOGY_DATASET for source in response.sources)


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
    looping = tool_turn((ToolName.SEARCH_MYTHOLOGY, {"query": "Thor"}))
    llm = ScriptedLLM([looping, looping, answer_turn("Thor is a god [S1].")])
    response = await make_agent(llm, retriever, max_tool_rounds=1).ask("Who is Thor?")

    assert llm.calls[-1]["tool_choice"] == "none"
    assert response.answer == "Thor is a god [S1]."


async def test_passages_from_same_section_share_one_source(retriever):
    """Two searches hitting the same section produce one source id, not duplicates."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_MYTHOLOGY, {"query": "Thor"}), (ToolName.SEARCH_MYTHOLOGY, {"query": "Mjolnir"})),
        answer_turn("Thor wields Mjölnir [S1]."),
    ])
    response = await make_agent(llm, retriever).ask("What does Thor wield?")
    assert [source.id for source in response.sources] == ["S1"]


async def test_lenticular_citations_are_normalized(retriever):
    """Drifting citation formats and invisible characters are normalized; ids still count."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_MYTHOLOGY, {"query": "Odin"})),
        answer_turn("Odin is wise【S1】, old [ S1 ] and one\u2011eyed【S1†L9-L12】 [\u200bS1]."),
    ])
    response = await make_agent(llm, retriever).ask("Who is Odin?")
    assert response.answer == "Odin is wise[S1], old [S1] and one-eyed[S1] [S1]."
    assert [source.id for source in response.sources] == ["S1"]


async def test_provider_specific_tool_call_fields_are_echoed_back(retriever):
    """Opaque tool-call fields (e.g. Gemini thought signatures) are sent back in the next request."""
    signature = {"extra_content": {"google": {"thought_signature": "sig"}}}
    call = ToolCall(id="c1", name=ToolName.SEARCH_MYTHOLOGY, arguments='{"query": "Odin"}', extra=signature)
    llm = ScriptedLLM([AssistantTurn(content=None, model="m", tool_calls=[call]), answer_turn("Odin [S1].")])
    await make_agent(llm, retriever).ask("Who is Odin?")

    assistant_message = llm.calls[1]["messages"][-2]
    assert assistant_message["tool_calls"][0]["extra_content"] == signature["extra_content"]


async def test_repeated_identical_tool_call_is_short_circuited(retriever):
    """The same call twice in one request is not re-executed; the model is told to use the first result."""
    search = (ToolName.SEARCH_MYTHOLOGY, {"query": "Odin"})
    llm = ScriptedLLM([tool_turn(search), tool_turn(search), answer_turn("Odin is wise [S1].")])
    await make_agent(llm, retriever).ask("Who is Odin?")
    assert llm.calls[2]["messages"][-1]["content"] == DUPLICATE_CALL


async def test_empty_final_answer_is_retried_once(retriever):
    """An empty final message triggers one more call with tools disabled."""
    llm = ScriptedLLM([
        tool_turn((ToolName.SEARCH_MYTHOLOGY, {"query": "Odin"})),
        answer_turn(""),
        answer_turn("Odin is wise [S1]."),
    ])
    response = await make_agent(llm, retriever).ask("Who is Odin?")
    assert response.answer == "Odin is wise [S1]."
    assert llm.calls[-1]["tool_choice"] == "none"


async def test_missing_hero_points_the_model_to_mythology(retriever):
    """A failed superhero lookup tells the model mythology search is the other option."""
    llm = ScriptedLLM([tool_turn((ToolName.GET_SUPERHERO, {"name": "Achilles"})), answer_turn("...")])
    await make_agent(llm, retriever).ask("What was Achilles' weak spot?")
    assert ToolName.SEARCH_MYTHOLOGY in llm.calls[1]["messages"][-1]["content"]
