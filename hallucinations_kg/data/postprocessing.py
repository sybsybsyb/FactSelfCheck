from typing import Any

from hallucinations_kg.kg_builder.graph_restricter import GraphParser


def add_allowed_nodes_and_relationships(
    x: dict[str, Any], restrictions_column: str, response_sentences_column: str
) -> dict[str, Any]:
    parser = GraphParser()
    response_nodes, response_relationships = set(), set()
    for sentence_graph in x[f"{response_sentences_column}_raw_graph"]:
        graph = parser.parse(sentence_graph, postprocess=False)
        response_nodes.update([t[0] for t in graph] + [t[2] for t in graph])
        response_relationships.update([t[1] for t in graph])
    allowed_nodes = list(set(x[f"{restrictions_column}_entities"]) | response_nodes)
    allowed_relationships = list(
        set(x[f"{restrictions_column}_relations"]) | response_relationships
    )
    return {"allowed_nodes": allowed_nodes, "allowed_relationships": allowed_relationships}


def parse_graph(
    x: dict[str, Any],
    restrict: bool,
    samples_column: str,
    sentences_column: str,
    revert_postprocess: bool = False,
    postprocess: bool = True,
) -> dict[str, Any]:
    parser = GraphParser()
    result = {}
    for column in [samples_column, sentences_column]:
        triples, removed = [], []
        for sample in x[f"{column}_raw_graph"]:
            triples_, removed_ = parser(
                sample,
                x["allowed_nodes"],
                x["allowed_relationships"],
                restrict=restrict,
                revert_postprocess=revert_postprocess,
                postprocess=postprocess,
            )
            triples.append(triples_)
            removed.append(removed_)
            if column == sentences_column:
                assert len(removed_) == 0
        result.update({f"{column}_graph": triples, f"{column}_removed_triples": removed})
    return result


def postprocess_ents_rels(x: dict[str, Any], response_column: str) -> dict[str, Any]:
    parser = GraphParser()
    x[f"{response_column}_entities"] = [
        parser.postprocess(e) for e in x[f"{response_column}_entities"]
    ]
    x[f"{response_column}_relations"] = [
        parser.postprocess(r) for r in x[f"{response_column}_relations"]
    ]
    return x
