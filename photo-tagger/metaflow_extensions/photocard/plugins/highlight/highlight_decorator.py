from metaflow.decorators import StepDecorator


class OBHighlighterStepDecorator(StepDecorator):
    name = "ob_highlighter"

    def step_init(
        self, flow, graph, step_name, decorators, environment, flow_datastore, logger
    ):
        from metaflow.plugins.cards.card_decorator import CardDecorator

        try:
            [self._card] = [
                d
                for d in decorators
                if isinstance(d, CardDecorator) and d.attributes["type"] == "highlight"
            ]
        except:
            self._card = None

    def task_finished(
        self, step_name, flow, graph, is_task_ok, retry_count, max_user_code_retries
    ):
        from metaflow.metadata_provider import MetaDatum

        if self._card:
            tags = [f"attempt_id:{retry_count}"]
            self._add_metadata(
                [
                    MetaDatum(
                        field="highlight_card",
                        type="highlight_card",
                        value=self._card._card_uuid,
                        tags=tags,
                    )
                ]
            )

    def task_pre_step(
        self,
        step_name,
        task_datastore,
        metadata,
        run_id,
        task_id,
        flow,
        graph,
        retry_count,
        max_user_code_retries,
        ubf_context,
        inputs,
    ):
        self._add_metadata = lambda x: metadata.register_metadata(
            run_id, step_name, task_id, x
        )
