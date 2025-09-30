class GraphParser:
    def __call__(
        self,
        raw_graph: str,
        allowed_entities: list[str] | None,
        allowed_relations: list[str] | None,
        restrict: bool,
        revert_postprocess: bool = False,
        postprocess: bool = True,
    ) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
        triples = self.parse(raw_graph, postprocess)
        if restrict:
            assert allowed_entities is not None
            assert allowed_relations is not None
            revert_postprocess_entities = {self.postprocess(e): e for e in allowed_entities}
            revert_postprocess_relations = {self.postprocess(r): r for r in allowed_relations}

            allowed_entities = list(revert_postprocess_entities.keys())
            allowed_relations = list(revert_postprocess_relations.keys())
            restricted_triples, removed_triples = self.restrict(
                triples, allowed_entities, allowed_relations
            )
            if revert_postprocess:
                restricted_triples = [
                    (
                        revert_postprocess_entities[h],
                        revert_postprocess_relations[r],
                        revert_postprocess_entities[t],
                    )
                    for h, r, t in restricted_triples
                ]
            return restricted_triples, removed_triples
        else:
            return triples, []

    def parse(self, raw_graph: str, postprocess: bool = True) -> list[tuple[str, str, str]]:
        triples = []
        for line in raw_graph.split("\n"):
            splited = line.split(";")
            if len(splited) != 3:
                continue
            h, r, t = splited
            if h == "subject" and r == "relation" and t == "object":
                continue
            if postprocess:
                h, r, t = self.postprocess(h), self.postprocess(r), self.postprocess(t)
            else:
                h, r, t = h.strip(), r.strip(), t.strip()
            triples.append((h, r, t))
        return triples

    def restrict(
        self,
        triples: list[tuple[str, str, str]],
        allowed_entities: list[str],
        allowed_relations: list[str],
    ) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
        restricted_triples, removed_triples = [], []
        for triple in triples:
            if (
                triple[0] in allowed_entities
                and triple[1] in allowed_relations
                and triple[2] in allowed_entities
            ):
                restricted_triples.append(triple)
            else:
                removed_triples.append(triple)
        return restricted_triples, removed_triples

    def postprocess(self, name: str) -> str:
        name = name.strip().lower()
        name = name.replace(" ", "_")
        return name
