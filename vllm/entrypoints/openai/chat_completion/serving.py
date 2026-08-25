# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from copy import copy
from http import HTTPStatus
from typing import Any, Final, cast

from fastapi import Request

from vllm.config.reasoning import DynamicEffortConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
    make_tool_call_id,
)
from vllm.entrypoints.generate.base.serving import (
    GenerateBaseServing,
    GenerationError,
    build_per_request_timing_metrics,
    clamp_prompt_logprobs,
    format_token_id_placeholder,
)
from vllm.entrypoints.openai.chat_completion.dynamic_effort import (
    DynamicEffortError,
    EffortRender,
    EffortVariant,
    EffortVariants,
    apply_default_effort,
    apply_dynamic_effort,
    split_body_and_tails,
)
from vllm.entrypoints.openai.chat_completion.effort_tails import (
    tokenize_variant_tails,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    EffortInfo,
)
from vllm.entrypoints.openai.engine.protocol import (
    CompletionTokenUsageInfo,
    DeltaMessage,
    ErrorResponse,
    FunctionCall,
    PerRequestTimingMetrics,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens, should_include_usage
from vllm.entrypoints.serve.utils.request_logger import RequestLogger
from vllm.entrypoints.serve.utils.tool_calls_utils import (
    maybe_filter_parallel_tool_calls,
)
from vllm.inputs import EngineInput, MultiModalPlaceholders
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import RequestOutput
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import Parser
from vllm.renderers import merge_kwargs
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.utils.collection_utils import as_list
from vllm.utils.mistral import is_mistral_tokenizer
from vllm.utils.serial_utils import numpy2base64

logger = init_logger(__name__)


def _get_mm_token_counts(engine_input: EngineInput) -> dict[str, int]:
    """Sum per-modality placeholder tokens from ``mm_placeholders``.

    Keyed by modality name; ``PlaceholderRange.length`` is the placeholder's
    prompt token span, so each sum matches the placeholder tokens already
    counted in ``usage.prompt_tokens``.
    """
    mm_placeholders = cast(
        MultiModalPlaceholders | None, engine_input.get("mm_placeholders")
    )
    return {
        modality: sum(p.length for p in ranges)
        for modality, ranges in (mm_placeholders or {}).items()
        if ranges
    }


def _make_completion_tokens_details(
    parser: "Parser | None", output_token_ids: list[GenericSequence[int]]
) -> CompletionTokenUsageInfo | None:
    """`completion_tokens_details.reasoning_tokens` from the reasoning parser's
    span count over each choice's generated ids; None without a parser."""
    if parser is None or parser.reasoning_parser is None:
        return None
    count = parser.reasoning_parser.count_reasoning_tokens
    return CompletionTokenUsageInfo(
        reasoning_tokens=sum(count(ids) for ids in output_token_ids)
    )


def _make_prompt_tokens_details(
    enable_prompt_tokens_details: bool,
    num_cached_tokens: int | None,
    num_cache_creation_tokens: int | None,
    mm_token_counts: dict[str, int] | None,
) -> PromptTokenUsageInfo | None:
    """Build ``prompt_tokens_details`` from cached + multimodal token counts."""
    if not enable_prompt_tokens_details:
        return None
    if (
        num_cached_tokens is None
        and num_cache_creation_tokens is None
        and not mm_token_counts
    ):
        return None
    return PromptTokenUsageInfo(
        cached_tokens=num_cached_tokens,
        created_cache_tokens=num_cache_creation_tokens,
        multimodal_tokens=mm_token_counts or None,
    )


class OpenAIServingChat(GenerateBaseServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        online_renderer: "OnlineRenderer",
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        enable_log_deltas: bool = True,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        enable_per_request_metrics: bool = False,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.online_renderer = online_renderer
        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
        self.enable_log_outputs = enable_log_outputs
        self.enable_log_deltas = enable_log_deltas

        self.enable_auto_tools: bool = enable_auto_tools
        self.parser_cls = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=self.model_config.model,
            is_harmony=self.model_config.hf_config.model_type == "gpt_oss",
        )
        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage
        self.enable_per_request_metrics = enable_per_request_metrics
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        mc = self.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )
        # NOTE(woosuk): While OpenAI's chat completion API supports browsing
        # for some models, currently vLLM doesn't support it. Please use the
        # Responses API instead.
        self.supports_browsing = False
        self.browser_tool = None
        # NOTE(woosuk): Chat completion API does not support code interpreter.
        # Please use the Responses API instead.
        self.supports_code_interpreter = False
        self.python_tool = None
        self._effort_pieces: dict[tuple, str] = {}

    def _dynamic_effort_config(self) -> DynamicEffortConfig | None:
        vllm_config = getattr(self.engine_client, "vllm_config", None)
        reasoning_config = getattr(vllm_config, "reasoning_config", None)
        return getattr(reasoning_config, "dynamic_effort", None)

    def _effective_chat_template_kwargs(
        self, request: ChatCompletionRequest
    ) -> dict[str, Any]:
        return (
            request.build_chat_params(
                self.chat_template,
                self.chat_template_content_format,
            )
            .with_defaults(self.default_chat_template_kwargs)
            .chat_template_kwargs
        )

    async def render_chat_request(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse:
        """
        Validate the model and preprocess a chat completion request.

        Delegates preprocessing logic to OnlineRenderer, adding the
        engine-aware checks (LoRA model validation, engine health).

        Returns:
            A tuple of (conversation, engine_inputs) on success,
            or an ErrorResponse on failure.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        return await self.online_renderer.render_chat(request)

    async def _render_effort_variants_full(
        self,
        default_input: EngineInput,
        renders: list[EffortRender],
        default: EffortRender,
    ) -> list[list[int] | None] | ErrorResponse | None:
        """Token ids of every level, each through the full renderer.

        A render's insert and suffix are tokenized on their own and spliced
        around the generation prompt's ids; a render with no request starts
        from the default level's ids. `None` when the pieces cannot be peeled
        off the ids exactly."""
        tokenizer = self.renderer.tokenizer
        if tokenizer is None:
            return None

        def encode(text: str) -> list[int]:
            return list(tokenizer.encode(text, add_special_tokens=False))

        def peel(ids: list[int], text: str) -> list[int] | None:
            tail = encode(text)
            if not text:
                return ids
            if not tail or ids[-len(tail) :] != tail:
                return None
            return ids[: -len(tail)]

        def compose(template_ids: list[int], render: EffortRender) -> list[int] | None:
            ids = template_ids
            if render.insert:
                ids = peel(ids, render.gen)
                if ids is None:
                    return None
                ids = ids + encode(render.insert + render.gen)
            return ids + encode(render.suffix)

        default_ids = self._extract_prompt_components(default_input).token_ids
        if default_ids is None:
            return None
        base_ids = peel(list(default_ids), default.suffix)
        if base_ids is not None and default.insert:
            base_ids = peel(base_ids, default.insert + default.gen)
            if base_ids is not None:
                base_ids += encode(default.gen)
        rendered: list[list[int] | None] = []
        for render in renders:
            if render.request is None:
                if render is default:
                    rendered.append(list(default_ids))
                    continue
                if base_ids is None:
                    return None
                rendered.append(compose(base_ids, render))
                continue
            result = await self.render_chat_request(render.request)
            if isinstance(result, ErrorResponse):
                return result
            _, variant_inputs = result
            if len(variant_inputs) != 1:
                return None
            ids = self._extract_prompt_components(variant_inputs[0]).token_ids
            rendered.append(None if ids is None else compose(list(ids), render))
        return rendered

    async def _tokenize_effort_variants(
        self,
        request: ChatCompletionRequest,
        default_input: EngineInput,
        renders: list[EffortRender],
        default: EffortRender,
    ) -> list[list[int]] | None:
        """Token ids of every level from one tokenization of the conversation.

        Each non-default level is run through the chat template only (no
        tokenization), or reuses the default level's template output when only
        its insert or suffix differs; its ids are the default level's ids up
        to the last special token the rendered texts share, plus the tokenized
        remainder. Texts that diverge earlier than the last message (e.g.
        `preserve_thinking=false` with a think-off level) just get a longer
        tail. `None` when exactness cannot be proven - multimodal or truncated
        prompts, a non-HF tokenizer, no shared special token, or a default
        tail that does not re-tokenize to its own ids.
        """
        tokenizer = self.renderer.tokenizer
        if (
            tokenizer is None
            or is_mistral_tokenizer(tokenizer)
            or self.model_config.enable_prompt_embeds
            or request.truncate_prompt_tokens is not None
        ):
            return None
        components = self._extract_prompt_components(default_input)
        default_text, default_ids = components.text, components.token_ids
        if (
            not isinstance(default_text, str)
            or default_ids is None
            or not isinstance(default_input, dict)
            or "multi_modal_data" in default_input
        ):
            return None
        base_text = default.base_text(default_text)
        if base_text is None:
            return None
        variant_texts: list[str] = []
        for render in renders:
            if render is default:
                variant_texts.append(default_text)
                continue
            text = (
                base_text
                if render.request is None
                else await self._render_effort_variant_text(render.request)
            )
            composed = None if text is None else render.compose(text)
            if composed is None:
                return None
            variant_texts.append(composed)
        return tokenize_variant_tails(
            lambda text: tokenizer.encode(text, add_special_tokens=False),
            default_text,
            default_ids,
            variant_texts,
            tokenizer.all_special_tokens,
        )

    async def _render_effort_variant_text(
        self, request: ChatCompletionRequest
    ) -> str | None:
        """The chat template's output for `request`, exactly as
        `OnlineRenderer.preprocess_chat` would build it, but not tokenized."""
        online = self.online_renderer
        if request.tools is None or (
            request.tool_choice == "none" and online.exclude_tools_when_tool_choice_none
        ):
            tool_dicts = None
        else:
            tool_dicts = [tool.model_dump() for tool in request.tools]
        mm_config = self.model_config.multimodal_config
        chat_params = request.build_chat_params(
            online.chat_template, online.chat_template_content_format
        ).with_defaults(
            merge_kwargs(
                online.default_chat_template_kwargs,
                dict(tools=tool_dicts, tokenize=False),
            ),
            default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
            default_mm_processor_kwargs=request.mm_processor_kwargs,
        )
        _, prompt = await online.renderer.render_messages_async(
            request.messages, chat_params
        )
        if not isinstance(prompt, dict) or "multi_modal_data" in prompt:
            return None
        text = prompt.get("prompt")
        return text if isinstance(text, str) else None

    def _effort_piece_key(self, request: ChatCompletionRequest) -> tuple:
        kwargs = request.chat_template_kwargs or {}
        return (
            request.chat_template,
            request.reasoning_effort,
            tuple(sorted((k, repr(v)) for k, v in kwargs.items())),
        )

    async def _effort_generation_prompt(
        self, request: ChatCompletionRequest, rendered_text: str
    ) -> str | None:
        """The chat template's generation prompt for `request`: what
        `add_generation_prompt=True` adds after the last message. Cached per
        template and kwargs; a cached value is trusted only when the text
        actually ends with it, otherwise it is re-derived from two renders."""
        key = ("gen", *self._effort_piece_key(request))
        cached = self._effort_pieces.get(key)
        if cached and rendered_text.endswith(cached):
            return cached
        if not request.add_generation_prompt or request.continue_final_message:
            return None
        without = copy(request)
        without.add_generation_prompt = False
        text = await self._render_effort_variant_text(without)
        if text is None or not rendered_text.startswith(text):
            return None
        gen = rendered_text[len(text) :]
        if not gen:
            return None
        self._effort_pieces[key] = gen
        return gen

    async def _effort_system_turn(
        self, request: ChatCompletionRequest, sentence: str
    ) -> str | None:
        """The chat template's rendering of one system message carrying
        `sentence`, as it would appear at the head of a conversation. A
        template that will not render it as a clean single turn (a default
        system prompt, tools, a rejected lone system message) yields `None`.
        Assembled as text because the template rejects a system message that
        is not first."""
        key = ("system", sentence, *self._effort_piece_key(request))
        cached = self._effort_pieces.get(key)
        if cached is not None:
            return cached or None
        probe = "effort placement probe"
        with_system = copy(request)
        with_system.add_generation_prompt = False
        with_system.tools = None
        with_system.messages = [
            {"role": "system", "content": sentence},
            {"role": "user", "content": probe},
        ]
        without_system = copy(with_system)
        without_system.messages = [{"role": "user", "content": probe}]
        turn = None
        try:
            a = await self._render_effort_variant_text(with_system)
            b = await self._render_effort_variant_text(without_system)
        except Exception as exc:
            logger.debug("dynamic_effort: no system turn for the template: %s", exc)
            a = b = None
        if a and b and a.endswith(b):
            turn = a[: -len(b)]
            if not turn or probe in turn or sentence not in turn:
                turn = None
        self._effort_pieces[key] = turn or ""
        return turn

    async def _apply_default_effort_layout(
        self,
        request: ChatCompletionRequest,
        variants: EffortVariants,
        engine_inputs: list[EngineInput],
    ) -> None:
        """Apply the default level's system insert and suffix to the prompt
        already rendered, text and ids. A default level whose system turn
        cannot be derived is rendered as a trailing user message instead by
        `_effort_renders`, so here only the template pieces are spliced."""
        if len(engine_inputs) != 1:
            return
        variant = variants.levels[variants.default_level]
        engine_input = engine_inputs[0]
        tokenizer = self.renderer.tokenizer
        if (
            (not variant.system and not variant.suffix)
            or tokenizer is None
            or not isinstance(engine_input, dict)
        ):
            return
        text = engine_input.get("prompt")
        ids = engine_input.get("prompt_token_ids")
        if not isinstance(text, str):
            return
        render = await self._resolve_effort_render(request, variant, text)
        if render.request is not None:
            # Fallback: the level re-renders as a trailing user message.
            result = await self.render_chat_request(render.request)
            if isinstance(result, ErrorResponse) or len(result[1]) != 1:
                return
            engine_inputs[0] = result[1][0]
            request.messages = render.request.messages
            return
        composed = render.compose(text)
        if composed is None:
            return
        engine_input["prompt"] = composed  # type: ignore[typeddict-item]
        if ids is None:
            return
        new_ids: list[int] | None = list(ids)
        if render.insert:
            gen_ids = tokenizer.encode(render.gen, add_special_tokens=False)
            if new_ids[-len(gen_ids) :] == list(gen_ids):
                new_ids = new_ids[: -len(gen_ids)] + list(
                    tokenizer.encode(
                        render.insert + render.gen, add_special_tokens=False
                    )
                )
            else:
                new_ids = None
        if new_ids is None:
            new_ids = list(tokenizer.encode(composed, add_special_tokens=False))
        else:
            new_ids += list(tokenizer.encode(render.suffix, add_special_tokens=False))
        engine_input["prompt_token_ids"] = new_ids  # type: ignore[typeddict-item]

    async def _resolve_effort_render(
        self,
        request: ChatCompletionRequest,
        variant: EffortVariant,
        rendered_text: str,
        kwargs: dict[str, Any] | None = None,
    ) -> EffortRender:
        """The render of a variant that shares the default level's template
        output. A system sentence becomes an insert before the generation
        prompt; when either piece cannot be derived from the template the
        level falls back to the trailing-user-message rendering."""
        if kwargs is None:
            kwargs = dict(request.chat_template_kwargs or {})
        if variant.system:
            gen = await self._effort_generation_prompt(request, rendered_text)
            turn = (
                None
                if gen is None
                else await self._effort_system_turn(request, variant.system)
            )
            if gen is None or turn is None:
                fallback = copy(request)
                fallback.messages = variant.messages + [
                    {"role": "user", "content": variant.system}
                ]
                fallback.chat_template_kwargs = kwargs or None
                return EffortRender(fallback, suffix=variant.suffix)
            return EffortRender(None, insert=turn, gen=gen, suffix=variant.suffix)
        return EffortRender(None, suffix=variant.suffix)

    async def _effort_renders(
        self,
        request: ChatCompletionRequest,
        variants: EffortVariants,
        rendered_text: str,
    ) -> list[EffortRender]:
        """One render per level (plus the off-vote question last): the request
        to run through the chat template, or none when the level's template
        output is the default level's and only its insert or suffix differs."""
        default_level = variants.default_level
        default_variant = variants.levels[default_level]
        default_kwargs = dict(request.chat_template_kwargs or {})
        renders: list[EffortRender] = []
        for level, variant in enumerate(variants.levels):
            kwargs = dict(default_kwargs)
            if level == default_level:
                renders.append(
                    await self._resolve_effort_render(request, variant, rendered_text)
                )
                continue
            if level in variants.think_off_levels:
                kwargs["enable_thinking"] = False
            elif default_level in variants.think_off_levels:
                kwargs["enable_thinking"] = True
            if (
                variant.messages is default_variant.messages
                and kwargs == default_kwargs
            ):
                renders.append(
                    await self._resolve_effort_render(
                        request, variant, rendered_text, kwargs
                    )
                )
                continue
            variant_request = copy(request)
            variant_request.messages = variant.messages
            variant_request.chat_template_kwargs = kwargs or None
            renders.append(EffortRender(variant_request, suffix=variant.suffix))
        if variants.meta_messages is not None:
            meta_request = copy(request)
            meta_request.messages = variants.meta_messages
            meta_request.chat_template_kwargs = {
                **default_kwargs,
                "enable_thinking": False,
            }
            renders.append(EffortRender(meta_request))
        return renders

    async def _attach_effort_tails(
        self,
        request: ChatCompletionRequest,
        engine_inputs: list[EngineInput],
    ) -> ErrorResponse | None:
        """Render the other levels and record the §13.3 body/tail seam.

        The prompt already submitted is the default-level variant, its suffix
        included. The rest cost at most one chat-template pass each and
        tokenize only their tail; the engine only ever prefills the body once,
        because the body is identical across levels. Anything unexpected
        (multiple prompts, an empty seam) silently leaves the request on
        today's single-level path.
        """
        variants = request._dynamic_effort_variants
        request._dynamic_effort_variants = None
        if variants is None or request._dynamic_effort is None:
            return None
        if len(engine_inputs) != 1:
            return None
        think_off_levels = variants.think_off_levels
        has_meta = variants.meta_messages is not None
        default_text = self._extract_prompt_components(engine_inputs[0]).text
        if not isinstance(default_text, str):
            return None
        renders = await self._effort_renders(request, variants, default_text)
        default = renders[variants.default_level]
        if default.request is not None:
            return None
        rendered = await self._tokenize_effort_variants(
            request, engine_inputs[0], renders, default
        )
        if rendered is None:
            rendered = await self._render_effort_variants_full(
                engine_inputs[0], renders, default
            )
        if isinstance(rendered, ErrorResponse):
            return rendered
        if rendered is None or any(ids is None for ids in rendered):
            return None
        split = split_body_and_tails([list(ids) for ids in rendered])  # type: ignore[arg-type]
        if split is None:
            return None
        body_len, tails = split
        if has_meta:
            # tails[-1] is the hidden yes/no question; the engine samples it
            # `off_votes` times before it lets a think-off verdict stand.
            request._dynamic_effort["meta_tail"] = tails.pop()
            tokenizer = self.renderer.tokenizer
            stop_ids: set[int] = set()
            yes_ids: set[int] = set()
            no_ids: set[int] = set()
            if tokenizer is not None:
                for text in ("\n", "\n\n"):
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    if len(ids) == 1:
                        stop_ids.add(int(ids[0]))
                eos = getattr(tokenizer, "eos_token_id", None)
                if eos is not None:
                    stop_ids.add(int(eos))
                for text, bucket in (
                    ("yes", yes_ids),
                    ("Yes", yes_ids),
                    ("YES", yes_ids),
                    (" yes", yes_ids),
                    (" Yes", yes_ids),
                    ("no", no_ids),
                    ("No", no_ids),
                    ("NO", no_ids),
                    (" no", no_ids),
                    (" No", no_ids),
                ):
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    if len(ids) == 1:
                        bucket.add(int(ids[0]))
            request._dynamic_effort["meta_stop_ids"] = sorted(stop_ids)
            request._dynamic_effort["yes_ids"] = sorted(yes_ids)
            request._dynamic_effort["no_ids"] = sorted(no_ids)
        if think_off_levels and not default.suffix:
            # Thinking off is the default rendering plus a closed think block:
            # appended in place, no resubmission (measured identical to the
            # template's own `<think>\n\n</think>\n\n` rendering). Only valid
            # when the default prompt ends with the generation prompt: a
            # `think` placement default has its sentence after it, so the
            # engine resubmits the rendered think-off tail instead.
            tokenizer = self.renderer.tokenizer
            if tokenizer is not None:
                request._dynamic_effort["off_append"] = list(
                    tokenizer.encode("</think>\n\n", add_special_tokens=False)
                )
        request._dynamic_effort["body_len"] = body_len
        request._dynamic_effort["tails"] = tails
        return None

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        """
        return await self._with_kv_transfer_rejection_cleanup(
            self._create_chat_completion(request, raw_request), request, raw_request
        )

    async def _create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        # Streaming response
        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None
        requested_effort = request.reasoning_effort
        effort_config = self._dynamic_effort_config()
        # An omitted reasoning_effort takes the server's default before
        # anything else looks at it, so `dynamic` can be that default.
        apply_default_effort(request, effort_config)
        request._effort_request = {
            "requested": requested_effort,
            "effective": request.reasoning_effort,
        }
        if request.reasoning_effort == "dynamic":
            try:
                apply_dynamic_effort(request, effort_config)
            except DynamicEffortError as e:
                return self.create_error_response(str(e))
        chat_template_kwargs = self._effective_chat_template_kwargs(request)
        parser: Parser | None = None
        if self.parser_cls is not None:
            parser = self.parser_cls(
                tokenizer,
                request.tools,
                chat_template_kwargs=chat_template_kwargs,
                model_config=self.model_config,
            )
        result = await self.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            return result

        conversation, engine_inputs = result
        if request._dynamic_effort_variants is not None:
            await self._apply_default_effort_layout(
                request, request._dynamic_effort_variants, engine_inputs
            )
            variant_error = await self._attach_effort_tails(request, engine_inputs)
            if variant_error is not None:
                return variant_error

        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

        model_name = self.models.model_name(lora_request)

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        mm_token_counts: dict[str, int] | None = None
        for i, engine_input in enumerate(engine_inputs):
            prompt_token_ids = self._extract_prompt_components(engine_input).token_ids
            mm_token_counts = _get_mm_token_counts(engine_input)

            # If we are creating sub requests for multiple prompts, ensure that they
            # have unique request ids.
            sub_request_id = (
                request_id if len(engine_inputs) == 1 else f"{request_id}_{i}"
            )

            max_tokens = get_max_tokens(
                max_model_len,
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens,
                self._extract_prompt_len(engine_input),
                self.default_sampling_params,
                self.override_max_tokens,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
            )

            sampling_params: SamplingParams | BeamSearchParams
            if request.use_beam_search:
                sampling_params = request.to_beam_search_params(
                    max_tokens, self.default_sampling_params
                )
            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.default_sampling_params,
                )

            self._log_inputs(
                sub_request_id,
                engine_input,
                params=sampling_params,
                lora_request=lora_request,
            )

            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )
            session_id = self._get_session_id(request, raw_request)

            if isinstance(sampling_params, BeamSearchParams):
                generator = self.beam_search(
                    prompt=engine_input,
                    request_id=sub_request_id,
                    params=sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    session_id=session_id,
                )
            else:
                if not request.include_reasoning:
                    reasoning_ended = True
                elif request._grammar_from_parser:
                    # The Mistral grammar already includes an optional
                    # `think?` rule that handles both reasoning and
                    # non-reasoning outputs.
                    reasoning_ended = True
                elif parser is not None and parser.reasoning_parser is not None:
                    reasoning_ended = parser.is_reasoning_end(prompt_token_ids or [])
                else:
                    reasoning_ended = None

                generator = self.engine_client.generate(
                    engine_input,
                    sampling_params,
                    sub_request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=self._get_priority(request, raw_request),
                    data_parallel_rank=data_parallel_rank,
                    session_id=session_id,
                    reasoning_ended=reasoning_ended,
                    reasoning_parser_kwargs={
                        "chat_template_kwargs": chat_template_kwargs,
                    }
                    if parser is not None and parser.reasoning_parser is not None
                    else None,
                )

            generators.append(generator)

        assert len(generators) == 1
        (result_generator,) = generators

        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                chat_template_kwargs=chat_template_kwargs,
                mm_token_counts=mm_token_counts,
            )

        return await self.chat_completion_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            parser=parser,
            mm_token_counts=mm_token_counts,
        )

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    def _create_chat_message(self, *args: Any, **kwargs: Any) -> ChatMessage:
        """Construct the response :class:`ChatMessage` for the non-streaming path.

        The full-generator calls this at every construction site so
        subclasses can swap in a specialized :class:`ChatMessage`
        subclass (e.g. :class:`CohereServingChatV2` returning
        :class:`CohereChatMessage`) without duplicating the branchy
        tool-choice / auto-tools logic that decides which fields are
        populated. The default returns a plain :class:`ChatMessage`.
        """
        return ChatMessage(*args, **kwargs)

    def _finalize_response_message(
        self,
        message: ChatMessage,
        *,
        parser: Parser | None,
    ) -> ChatMessage:
        """Subclass hook to enrich a fully-constructed :class:`ChatMessage`.

        Default is a no-op. Subclasses that need to surface parser-side
        extras (e.g. :class:`CohereServingChatV2` reading grounding
        citations off the reasoning parser and populating
        :class:`CohereChatMessage.citations`) override this to inspect
        ``parser`` and mutate/replace ``message``.
        """
        return message

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        chat_template_kwargs: dict[str, Any] | None = None,
        mm_token_counts: dict[str, int] | None = None,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        previous_token_ids: list[list[int]] = [[] for _ in range(num_choices)]
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0
        num_cached_tokens = None
        num_cache_creation_tokens = None
        tools_streamed = [False] * num_choices

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        previous_texts = [""] * num_choices

        try:
            if self.parser_cls is not None:
                if tokenizer is None:
                    raise ValueError(
                        "Tokenizer not available when `skip_tokenizer_init=True`"
                    )
                parsers: list[Parser | None] = [
                    self.parser_cls(
                        tokenizer,
                        request.tools,
                        chat_template_kwargs=chat_template_kwargs,
                        model_config=self.model_config,
                    )
                    for _ in range(num_choices)
                ]
            else:
                parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in parser creation.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        stream_options = request.stream_options
        include_usage, include_continuous_usage = should_include_usage(
            stream_options, self.enable_force_include_usage
        )

        last_res: RequestOutput | None = None
        try:
            async for res in result_generator:
                last_res = res
                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    num_cache_creation_tokens = res.num_cache_creation_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # ``res.prompt`` is the rendered chat-templated prompt
                    prompt_text = res.prompt if request.return_prompt_text else None

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None,
                        )

                        # return prompt_token_ids at the first chunk ever
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            prompt_token_ids=(
                                res.prompt_token_ids
                                if request.return_token_ids
                                else None
                            ),
                            prompt_text=prompt_text,
                        )

                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens,
                            )

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: str | list[dict[str, str]] = ""
                        if (
                            conversation
                            and "content" in conversation[-1]
                            and conversation[-1].get("role") == role
                        ):
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(content=last_msg_content),
                                    logprobs=None,
                                    finish_reason=None,
                                )
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name,
                                )
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens,
                                    )

                                data = chunk.model_dump_json(exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    parser = parsers[i]
                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and (
                        request.top_logprobs is not None or request.logprob_token_ids
                    ):
                        assert output.logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                            logprob_token_ids=request.logprob_token_ids,
                            return_as_token_id=request.return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    delta_text = output.text

                    if (
                        not delta_text
                        and not output.token_ids
                        and not previous_num_tokens[i]
                    ):
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: DeltaMessage | None

                    if parser is not None:
                        delta_message = parser.parse_delta(
                            delta_text=delta_text,
                            delta_token_ids=as_list(output.token_ids),
                            request=request,
                            prompt_token_ids=res.prompt_token_ids,
                            finished=output.finish_reason is not None,
                        )
                        if delta_message is not None and delta_message.tool_calls:
                            tools_streamed[i] = True

                    # handle streaming just a content delta (no parsers)
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    previous_texts[i] += delta_text

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)
                    previous_token_ids[i].extend(output.token_ids)

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    # When reasoning is hidden, suppress per-token
                    # metadata (logprobs, token_ids) on every chunk to
                    # prevent leaking reasoning tokens through decoded
                    # token text in logprob entries or raw token IDs.
                    hide_stream_metadata = (
                        not request.include_reasoning and parser is not None
                    )
                    if hide_stream_metadata:
                        logprobs = None

                    if delta_message is None:
                        # NOTE: If return_token_ids is enabled, we still need to
                        # send a chunk with token_ids even if delta_message is None
                        # to ensure all tokens are included in the response
                        if output.finish_reason is None and (
                            not request.return_token_ids or hide_stream_metadata
                        ):
                            continue
                        delta_message = DeltaMessage()

                    # Log streaming delta if output logging is enabled
                    if self.enable_log_outputs and self.request_logger:
                        delta_content_parts = []
                        if delta_message.content:
                            delta_content_parts.append(delta_message.content)
                        if delta_message.reasoning:
                            reasoning = delta_message.reasoning
                            delta_content_parts.append(f"[reasoning: {reasoning}]")
                        if delta_message.tool_calls:
                            tool_args = "".join(
                                tc.function.arguments
                                for tc in delta_message.tool_calls
                                if tc.function and tc.function.arguments
                            )
                            if tool_args:
                                delta_content_parts.append(f"[tool_calls: {tool_args}]")

                        if delta_content_parts and self.enable_log_deltas:
                            delta_content = " ".join(delta_content_parts)
                            self.request_logger.log_outputs(
                                request_id=request_id,
                                outputs=delta_content,
                                output_token_ids=as_list(output.token_ids),
                                finish_reason=output.finish_reason,
                                is_streaming=True,
                                delta=True,
                            )

                    include_token_ids = (
                        request.return_token_ids and not hide_stream_metadata
                    )

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None,
                            token_ids=(
                                as_list(output.token_ids) if include_token_ids else None
                            ),
                        )

                    # if the model is finished generating
                    else:
                        # check for error finish reason and abort streaming
                        # finish_reason='error' indicates a retryable error
                        self._raise_if_error(output.finish_reason, request_id)

                        # Send the finish response for each request.n only once
                        # In OpenAI's API, when a tool is called, the
                        # finish_reason is:
                        # "tool_calls" for "auto" or "required" tool calls,
                        # and "stop" for named tool calls.
                        if tools_streamed[i] and not tool_choice_function_name:
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = (
                                output.finish_reason if output.finish_reason else "stop"
                            )
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=finish_reason_,
                            stop_reason=output.stop_reason,
                            token_ids=(
                                as_list(output.token_ids) if include_token_ids else None
                            ),
                        )

                        finish_reason_sent[i] = True

                    choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )
                    # Stamp the fingerprint on terminal chunks only (those with
                    # finish_reason set). When ``include_usage`` is on, the
                    # trailing usage chunk below overrides this as the true
                    # final message.
                    if (
                        not include_usage
                        and self.system_fingerprint is not None
                        and choice_data.finish_reason is not None
                    ):
                        chunk.system_fingerprint = self.system_fingerprint

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                final_usage.prompt_tokens_details = _make_prompt_tokens_details(
                    self.enable_prompt_tokens_details,
                    num_cached_tokens,
                    num_cache_creation_tokens,
                    mm_token_counts,
                )
                final_usage.completion_tokens_details = _make_completion_tokens_details(
                    parsers[0] if parsers else None, previous_token_ids
                )

                # In streaming, metrics ride on this final usage chunk, which is
                # only emitted when usage reporting is enabled (i.e.
                # ``stream_options.include_usage=true`` or
                # ``--enable-force-include-usage``).
                stream_per_request_metrics: PerRequestTimingMetrics | None = None
                if (
                    self.enable_per_request_metrics
                    # See note in chat_completion_full_generator: suppress for n>1.
                    and (request.n or 1) == 1
                ):
                    last_metrics = last_res.metrics if last_res is not None else None
                    stream_per_request_metrics = build_per_request_timing_metrics(
                        last_metrics, completion_tokens
                    )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                    system_fingerprint=self.system_fingerprint,
                    metrics=stream_per_request_metrics,
                    effort=EffortInfo.from_report(
                        last_res.effort if last_res is not None else None
                    ),
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                # Log the complete response for each choice
                for i in range(num_choices):
                    full_text = (
                        previous_texts[i]
                        if previous_texts and i < len(previous_texts)
                        else f"<streaming_complete: {previous_num_tokens[i]} tokens>"
                    )
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=full_text,
                        output_token_ids=None,  # Consider also logging all token IDs
                        finish_reason="streaming_complete",
                        is_streaming=True,
                        delta=False,
                    )

        except GenerationError as e:
            yield f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
        except Exception as e:
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        parser: Parser | None = None,
        mm_token_counts: dict[str, int] | None = None,
    ) -> ErrorResponse | ChatCompletionResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        if final_res is None:
            return self.create_error_response(
                "No output received from the engine.",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        choices: list[ChatCompletionResponseChoice] = []

        role = self.get_chat_request_role(request)
        tool_parser_cls = (
            self.parser_cls.tool_parser_cls if self.parser_cls is not None else None
        )
        for output in final_res.outputs:
            # check for error finish reason and raise GenerationError
            # finish_reason='error' indicates a retryable request-level internal error
            self._raise_if_error(output.finish_reason, request_id)
            token_ids = output.token_ids
            out_logprobs = output.logprobs

            if request.logprobs and (
                request.top_logprobs is not None or request.logprob_token_ids
            ):
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    logprob_token_ids=request.logprob_token_ids,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            if parser is not None:
                reasoning, content, tool_calls = parser.parse(
                    output.text,
                    request,
                    enable_auto_tools=self.enable_auto_tools,
                    model_output_token_ids=token_ids,
                )
                suppress_metadata = not request.include_reasoning and parser is not None
                if not request.include_reasoning:
                    reasoning = None
                if suppress_metadata:
                    logprobs = None
            else:
                reasoning = None
                content = output.text
                tool_calls = []
                suppress_metadata = False

            auto_tools_called = False
            is_named_tool_choice = (
                request.tool_choice is not None
                and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
            )
            is_required_tool_choice = request.tool_choice == "required"

            # All six construction sites route through ``self._create_chat_message``
            # so subclasses can swap in a specialized :class:`ChatMessage`
            # (e.g. the Cohere v2 handler's ``CohereChatMessage``) without
            # having to duplicate this branch logic.
            if (not self.enable_auto_tools or not tool_parser_cls) and (
                not is_named_tool_choice and not is_required_tool_choice
            ):
                message = self._create_chat_message(
                    role=role, reasoning=reasoning, content=content
                )

            elif is_named_tool_choice or is_required_tool_choice:
                message = self._create_chat_message(
                    role=role,
                    reasoning=reasoning,
                    content=content or "",
                    tool_calls=[
                        ToolCall(id=tc.id or make_tool_call_id(), function=tc)
                        for tc in (tool_calls or [])
                    ],
                )

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":
                message = self._create_chat_message(
                    role=role, reasoning=reasoning, content=content
                )

            # handle when there are tools and tool choice is auto
            elif (
                request.tools
                and (request.tool_choice == "auto" or request.tool_choice is None)
                and self.enable_auto_tools
                and tool_parser_cls
            ):
                auto_tools_called = tool_calls is not None and len(tool_calls) > 0
                if tool_calls:
                    message = self._create_chat_message(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                        tool_calls=[
                            ToolCall(id=tc.id or make_tool_call_id(), function=tc)
                            for tc in tool_calls
                        ],
                    )

                else:
                    message = self._create_chat_message(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                    )

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion."
                )
                message = self._create_chat_message(
                    role=role, reasoning=reasoning, content=content
                )

            # Subclass hook: enrich the constructed message with any
            # parser-side extras that don't fit through the plain
            # ``(reasoning, content, tool_calls)`` tuple. Base is a no-op;
            # citation-aware handlers use this to surface grounding
            # metadata cached on the reasoning parser.
            message = self._finalize_response_message(message, parser=parser)

            # In OpenAI's API, when a tool is called, the finish_reason is:
            # "tool_calls" for "auto" or "required" tool calls,
            # and "stop" for named tool calls.
            is_finish_reason_tool_calls = auto_tools_called or (
                request.tool_choice
                and request.tool_choice == "required"
                and output.finish_reason == "stop"
            )

            routed_experts_b64 = (
                numpy2base64(output.routed_experts)
                if output.routed_experts is not None
                else None
            )

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls"
                if is_finish_reason_tool_calls
                else output.finish_reason
                if output.finish_reason
                else "stop",
                stop_reason=output.stop_reason,
                token_ids=(
                    as_list(output.token_ids)
                    if request.return_token_ids and not suppress_metadata
                    else None
                ),
                routed_experts=routed_experts_b64,
            )
            choice_data = maybe_filter_parallel_tool_calls(choice_data, request)

            choices.append(choice_data)

        if request.echo:
            last_msg_content: str | list[dict[str, str]] = ""
            if (
                conversation
                and "content" in conversation[-1]
                and conversation[-1].get("role") == role
            ):
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg["text"] for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs
        )
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        usage.prompt_tokens_details = _make_prompt_tokens_details(
            self.enable_prompt_tokens_details,
            final_res.num_cached_tokens,
            final_res.num_cache_creation_tokens,
            mm_token_counts,
        )
        usage.completion_tokens_details = _make_completion_tokens_details(
            parser, [output.token_ids for output in final_res.outputs]
        )

        request_metadata.final_usage_info = usage

        per_request_metrics: PerRequestTimingMetrics | None = None
        if (
            self.enable_per_request_metrics
            # Timing metrics describe a single generation stream. For n>1 the
            # returned stats belong to only one of the n sequences, so they
            # cannot be accurately attributed to the request; suppress instead.
            and (request.n or 1) == 1
        ):
            per_request_metrics = build_per_request_timing_metrics(
                final_res.metrics, num_generated_tokens
            )

        # ``final_res.prompt`` is the rendered chat-templated prompt text
        prompt_text = final_res.prompt if request.return_prompt_text else None

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            system_fingerprint=self.system_fingerprint,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            prompt_token_ids=(
                final_res.prompt_token_ids if request.return_token_ids else None
            ),
            prompt_text=prompt_text,
            kv_transfer_params=final_res.kv_transfer_params,
            ec_transfer_params=final_res.ec_transfer_params,
            metrics=per_request_metrics,
            effort=EffortInfo.from_report(final_res.effort),
        )

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                output_text = ""
                if choice.message.content:
                    output_text = choice.message.content
                elif choice.message.tool_calls:
                    # For tool calls, log the function name and arguments
                    tool_call_descriptions = []
                    for tc in choice.message.tool_calls:  # type: ignore
                        function_call: FunctionCall = tc.function  # type: ignore
                        tool_call_descriptions.append(
                            f"{function_call.name}({function_call.arguments})"
                        )
                    tool_calls_str = ", ".join(tool_call_descriptions)
                    output_text = f"[tool_calls: {tool_calls_str}]"

                if output_text:
                    # Get the corresponding output token IDs
                    output_token_ids = None
                    if choice.index < len(final_res.outputs):
                        output_token_ids = final_res.outputs[choice.index].token_ids

                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=output_text,
                        output_token_ids=output_token_ids,
                        finish_reason=choice.finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

        return response

    def _get_top_logprobs(
        self,
        logprobs: dict[int, Logprob],
        top_logprobs: int | None,
        tokenizer: TokenizerLike | None,
        should_return_as_token_id: bool,
        return_all: bool = False,
    ) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=(
                    token := self._get_decoded_token(
                        p[1],
                        p[0],
                        tokenizer,
                        return_as_token_id=should_return_as_token_id,
                    )
                ),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")),
            )
            for i, p in enumerate(logprobs.items())
            if return_all
            or top_logprobs == -1
            or (top_logprobs is not None and i < top_logprobs)
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        tokenizer: TokenizerLike | None,
        num_output_top_logprobs: int | None = None,
        logprob_token_ids: list[int] | None = None,
        return_as_token_id: bool | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                if should_return_as_token_id:
                    token = format_token_id_placeholder(token_id)
                else:
                    if tokenizer is None:
                        raise ValueError(
                            "Unable to get tokenizer because `skip_tokenizer_init=True`"
                        )

                    token = tokenizer.decode(token_id)

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    )
                )
            else:
                step_token = step_top_logprobs[token_id]
                step_decoded = step_token.decoded_token

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self._get_decoded_token(
                            step_token,
                            token_id,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                        logprob=max(step_token.logprob, -9999.0),
                        bytes=(
                            None
                            if step_decoded is None
                            else list(step_decoded.encode("utf-8", errors="replace"))
                        ),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs,
                            num_output_top_logprobs,
                            tokenizer,
                            should_return_as_token_id,
                            return_all=bool(logprob_token_ids),
                        ),
                    )
                )

        return ChatCompletionLogProbs(content=logprobs_content)
