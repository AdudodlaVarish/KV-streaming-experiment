# Native vLLM Apache-2.0 scheduler body with SLAI phase/victim adaptation.
def schedule(self, throttle_prefills: bool=False) -> SchedulerOutput:
    self.current_step += 1
    scheduled_new_reqs: list[Request] = []
    scheduled_resumed_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    preempted_reqs: list[Request] = []
    req_to_new_blocks: dict[str, KVCacheBlocks] = {}
    num_scheduled_tokens: dict[str, int] = {}
    token_budget = self.max_num_scheduled_tokens
    spec = self.vllm_config.speculative_config
    draft_slots = spec.max_num_new_slots_for_drafting if spec is not None else 0
    input_budget = self.scheduler_config.max_num_batched_tokens
    if self._pause_state == PauseState.PAUSED_ALL:
        token_budget = 0
    scheduled_encoder_inputs: dict[str, list[int]] = {}
    encoder_compute_budget = self.max_num_encoder_input_tokens
    scheduled_spec_decode_tokens: dict[str, list[int]] = {}
    prefill_scheduled = False
    has_sync_kv_loads = False
    scheduled_timestamp = time.monotonic()
    self.kv_cache_manager.new_step_starts()
    defer_prefills = (throttle_prefills and (not self.prefill_capacity_bound)) and any((not r.is_prefill_chunk for r in self.running))
    req_index = 0
    lab_deferred = self._slai_deferred()
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]
        if request.request_id in lab_deferred:
            req_index += 1
            continue
        if input_budget <= draft_slots:
            break
        if request.num_output_placeholders > 0 and request.num_computed_tokens + 2 - request.num_output_placeholders >= request.num_prompt_tokens + request.max_tokens:
            req_index += 1
            continue
        if self.current_step < request.next_decode_eligible_step:
            req_index += 1
            continue
        if defer_prefills and request.is_prefill_chunk:
            req_index += 1
            continue
        if self.ec_connector is not None and request.mm_features and (not self.ec_connector.ensure_cache_available(request, request.num_computed_tokens - request.num_output_placeholders)):
            req_index += 1
            continue
        num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)
        num_new_tokens = min(num_new_tokens, self.max_model_len - request.num_computed_tokens - self.num_sampled_tokens_per_step)
        if self.need_mamba_block_aligned_split:
            num_new_tokens = self._mamba_block_aligned_split(request, num_new_tokens)
        encoder_inputs_to_schedule = None
        external_load_encoder_input: list[int] = []
        new_encoder_compute_budget = encoder_compute_budget
        if request.has_encoder_inputs:
            encoder_inputs_to_schedule, num_new_tokens, new_encoder_compute_budget, external_load_encoder_input = self._try_schedule_encoder_inputs(request, request.num_computed_tokens, num_new_tokens, encoder_compute_budget, shift_computed_tokens=self.num_prefill_lookahead)
        num_new_tokens = self._reserve_prefill_lookahead(request, request.num_computed_tokens, num_new_tokens)
        if num_new_tokens == 0:
            req_index += 1
            continue
        with record_function_or_nullcontext('schedule: allocate_slots'):
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is not None:
                    break
                if self.connector is not None and self.connector.has_pending_block_frees():
                    break
                if self.policy == SchedulingPolicy.PRIORITY:
                    preempted_req = self._slai_victim(request, scheduled_new_reqs + scheduled_resumed_reqs + scheduled_running_reqs)
                else:
                    preempted_req = self.running[-1]
                if not self._request_blocks_can_be_freed(preempted_req):
                    break
                if self.policy == SchedulingPolicy.PRIORITY:
                    victim_index = self.running.index(preempted_req)
                    del self.running[victim_index]
                    if victim_index < req_index:
                        req_index -= 1
                    if preempted_req in scheduled_running_reqs:
                        preempted_req_id = preempted_req.request_id
                        scheduled_running_reqs.remove(preempted_req)
                        restored = num_scheduled_tokens.pop(preempted_req_id)
                        token_budget += restored
                        input_budget += restored + draft_slots
                        req_to_new_blocks.pop(preempted_req_id)
                        scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                        preempted_encoder_inputs = scheduled_encoder_inputs.pop(preempted_req_id, None)
                        if preempted_encoder_inputs:
                            num_embeds_to_restore = sum((preempted_req.get_num_encoder_embeds(i) for i in preempted_encoder_inputs))
                            encoder_compute_budget += num_embeds_to_restore
                else:
                    preempted_req = self.running.pop()
                self._preempt_request(preempted_req, scheduled_timestamp, drop_stale_output=self.requires_kv_delivery)
                preempted_reqs.append(preempted_req)
                if preempted_req == request:
                    break
        if new_blocks is None:
            break
        scheduled_running_reqs.append(request)
        prefill_scheduled |= request.is_prefill_chunk
        request_id = request.request_id
        req_to_new_blocks[request_id] = new_blocks
        num_scheduled_tokens[request_id] = num_new_tokens
        token_budget -= num_new_tokens
        input_budget -= num_new_tokens + draft_slots
        req_index += 1
        if request.spec_token_ids:
            num_scheduled_spec_tokens = num_new_tokens + request.num_computed_tokens - request.num_tokens - request.num_output_placeholders
            if num_scheduled_spec_tokens > 0:
                spec_token_ids = request.spec_token_ids
                if len(spec_token_ids) > num_scheduled_spec_tokens:
                    spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
            request.spec_token_ids = []
        if encoder_inputs_to_schedule:
            scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
            for i in encoder_inputs_to_schedule:
                self.encoder_cache_manager.allocate(request, i)
                if self.ec_connector is not None:
                    self.ec_connector.update_state_after_alloc(request, i)
            encoder_compute_budget = new_encoder_compute_budget
        if external_load_encoder_input:
            for i in external_load_encoder_input:
                self.encoder_cache_manager.allocate(request, i)
                if self.ec_connector is not None:
                    self.ec_connector.update_state_after_alloc(request, i)
    scheduled_loras: set[int] = set()
    if self.lora_config:
        scheduled_loras = set((req.lora_request.lora_int_id for req in scheduled_running_reqs if req.lora_request and req.lora_request.lora_int_id > 0))
        assert len(scheduled_loras) <= self.lora_config.max_loras
    if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
        step_skipped_waiting = create_request_queue(self.policy)
        while (self.waiting or self.skipped_waiting) and token_budget > 0:
            if input_budget <= draft_slots:
                break
            num_running = len(self.running) + self.num_waiting_for_streaming_input
            if num_running >= self.max_num_running_reqs:
                break
            request_queue = self._select_waiting_queue_for_scheduling()
            assert request_queue is not None
            request = request_queue.peek_request()
            request_id = request.request_id
            if self._is_blocked_waiting_status(request.status) and (not self._try_promote_blocked_waiting_request(request)):
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    logger.debug('%s is still in WAITING_FOR_REMOTE_KVS state.', request_id)
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
                continue
            if request.num_stale_output_tokens > 0 and (not request.drop_stale_output):
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
                continue
            if self.lora_config and request.lora_request and (len(scheduled_loras) == self.lora_config.max_loras and request.lora_request.lora_int_id not in scheduled_loras):
                request_queue.pop_request()
                step_skipped_waiting.prepend_request(request)
                continue
            num_external_computed_tokens = 0
            load_kv_async = False
            connector_prefix_cache_queries, connector_prefix_cache_hits = (0, 0)
            did_prefix_cache_lookup = False
            if request.num_computed_tokens == 0:
                did_prefix_cache_lookup = True
                new_computed_blocks, num_new_local_computed_tokens, request.shared_prefix_boundary, hit_diverged = self._get_local_prefix_cache_hit(request)
                if self.connector is not None:
                    partial_tail = num_new_local_computed_tokens % self.block_size
                    block_aligned_local = num_new_local_computed_tokens - partial_tail
                    ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(request, block_aligned_local)
                    if ext_tokens is None:
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue
                    if partial_tail and ext_tokens > partial_tail:
                        new_computed_blocks = self.kv_cache_manager.truncate_computed_blocks(new_computed_blocks, block_aligned_local)
                        num_new_local_computed_tokens = block_aligned_local
                        num_external_computed_tokens = ext_tokens
                    elif partial_tail:
                        num_external_computed_tokens = 0
                        load_kv_async = False
                    else:
                        num_external_computed_tokens = ext_tokens
                    if hit_diverged and num_external_computed_tokens == 0:
                        new_computed_blocks, num_new_local_computed_tokens, request.shared_prefix_boundary = self.kv_cache_manager.get_computed_blocks(request)
                    connector_prefix_cache_queries = request.num_tokens - num_new_local_computed_tokens
                    connector_prefix_cache_hits = num_external_computed_tokens
                num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
                assert num_computed_tokens <= request.num_tokens
                if self._ec_transfer_pending(request, num_computed_tokens):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue
                if request.prefill_stats and request.num_preemptions <= 0:
                    assert num_computed_tokens <= request.num_prompt_tokens
                    request.prefill_stats.set(num_prompt_tokens=request.num_prompt_tokens, num_local_cached_tokens=num_new_local_computed_tokens, num_external_cached_tokens=num_external_computed_tokens)
            else:
                new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                num_new_local_computed_tokens = 0
                num_computed_tokens = request.num_computed_tokens
                if self._ec_transfer_pending(request, num_computed_tokens):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue
            encoder_inputs_to_schedule = None
            external_load_encoder_input = []
            new_encoder_compute_budget = encoder_compute_budget
            pad_spec_decode = False
            if load_kv_async:
                assert num_external_computed_tokens > 0
                num_new_tokens = 0
            elif defer_prefills and num_computed_tokens < request.num_tokens - 1:
                break
            else:
                request_token_budget = min(token_budget, input_budget - draft_slots)
                num_new_tokens = request.num_tokens - num_computed_tokens
                if (self.num_spec_tokens > 0 and self.dynamic_sd_lookup is None) and self.num_sampled_tokens_per_step > 0 and (num_new_tokens == 1) and (not prefill_scheduled) and (scheduled_running_reqs or num_computed_tokens > 0):
                    padded_num_tokens = 1 + self.num_spec_tokens
                    if num_computed_tokens + padded_num_tokens + self.num_sampled_tokens_per_step <= self.max_model_len:
                        if padded_num_tokens > request_token_budget:
                            break
                        num_new_tokens = padded_num_tokens
                        pad_spec_decode = True
                threshold = self.scheduler_config.long_prefill_token_threshold
                if 0 < threshold < num_new_tokens:
                    num_new_tokens = threshold
                if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > request_token_budget:
                    break
                num_new_tokens = min(num_new_tokens, request_token_budget)
                assert num_new_tokens > 0
                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(request, num_new_tokens, num_new_local_computed_tokens, num_external_computed_tokens)
                    if num_new_tokens == 0:
                        break
                    if pad_spec_decode and num_new_tokens != 1 + self.num_spec_tokens:
                        num_new_tokens = 1
                        pad_spec_decode = False
                if request.has_encoder_inputs:
                    encoder_inputs_to_schedule, num_new_tokens, new_encoder_compute_budget, external_load_encoder_input = self._try_schedule_encoder_inputs(request, num_computed_tokens, num_new_tokens, encoder_compute_budget, shift_computed_tokens=self.num_prefill_lookahead)
                num_new_tokens = self._reserve_prefill_lookahead(request, num_computed_tokens, num_new_tokens)
                if num_new_tokens == 0:
                    break
            limit_lookahead_tokens = load_kv_async and self.num_lookahead_tokens > 0
            effective_lookahead_tokens = 0 if limit_lookahead_tokens else self.num_lookahead_tokens
            num_encoder_tokens = 0
            if self.is_encoder_decoder and request.has_encoder_inputs and encoder_inputs_to_schedule:
                num_encoder_tokens = sum((request.get_num_encoder_embeds(i) for i in encoder_inputs_to_schedule))
            reserved_blocks = 0
            if load_kv_async:
                reserved_blocks = self._inflight_prefill_reserved_blocks()
            new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, num_new_computed_tokens=num_new_local_computed_tokens, new_computed_blocks=new_computed_blocks, num_lookahead_tokens=effective_lookahead_tokens, num_external_computed_tokens=num_external_computed_tokens, delay_cache_blocks=load_kv_async, num_encoder_tokens=num_encoder_tokens, full_sequence_must_fit=self.scheduler_reserve_full_isl, reserved_blocks=reserved_blocks, has_scheduled_reqs=bool(self.running))
            if new_blocks is None:
                if request.has_encoder_inputs:
                    self.encoder_cache_manager.free(request)
                break
            if self.connector is not None:
                self.connector.update_state_after_alloc(request, self.kv_cache_manager.get_blocks(request_id), num_external_computed_tokens)
                if self.connector_prefix_cache_stats is not None and connector_prefix_cache_queries != 0:
                    self.connector_prefix_cache_stats.record(num_tokens=connector_prefix_cache_queries, num_hits=connector_prefix_cache_hits, preempted=request.num_preemptions > 0)
            if did_prefix_cache_lookup:
                self.kv_cache_manager.record_prefix_cache_stats(request, num_new_local_computed_tokens)
            request = request_queue.pop_request()
            if load_kv_async:
                request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                step_skipped_waiting.prepend_request(request)
                request.num_computed_tokens = num_computed_tokens
                self._inflight_prefills.add(request)
                if self.needs_kv_cache_zeroing:
                    self._skip_zero_block_ids.update(self.kv_cache_manager.get_zeroing_block_ids_in_range(request.request_id, num_new_local_computed_tokens, num_computed_tokens))
                continue
            self.running.append(request)
            if num_external_computed_tokens > 0:
                has_sync_kv_loads = True
            if self.log_stats:
                request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
            if request.status == RequestStatus.WAITING:
                scheduled_new_reqs.append(request)
            elif request.status == RequestStatus.PREEMPTED:
                scheduled_resumed_reqs.append(request)
            else:
                raise RuntimeError(f'Invalid request status: {request.status}')
            if self.lora_config and request.lora_request:
                scheduled_loras.add(request.lora_request.lora_int_id)
            req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(request_id)
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens = num_computed_tokens
            if pad_spec_decode:
                assert num_new_tokens == 1 + self.num_spec_tokens
                scheduled_spec_decode_tokens[request_id] = [-1] * self.num_spec_tokens
            if num_computed_tokens + num_new_tokens < request.num_tokens:
                self._inflight_prefills.add(request)
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
        if step_skipped_waiting:
            self.skipped_waiting.prepend_requests(step_skipped_waiting)
        if not defer_prefills:
            self.prefill_capacity_bound = bool(self.waiting)
    req_index = 0
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]
        if request.request_id not in lab_deferred:
            req_index += 1
            continue
        if input_budget <= draft_slots:
            break
        if request.num_output_placeholders > 0 and request.num_computed_tokens + 2 - request.num_output_placeholders >= request.num_prompt_tokens + request.max_tokens:
            req_index += 1
            continue
        if self.current_step < request.next_decode_eligible_step:
            req_index += 1
            continue
        if defer_prefills and request.is_prefill_chunk:
            req_index += 1
            continue
        if self.ec_connector is not None and request.mm_features and (not self.ec_connector.ensure_cache_available(request, request.num_computed_tokens - request.num_output_placeholders)):
            req_index += 1
            continue
        num_new_tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
        if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
            num_new_tokens = self.scheduler_config.long_prefill_token_threshold
        num_new_tokens = min(num_new_tokens, token_budget, input_budget - draft_slots)
        num_new_tokens = min(num_new_tokens, self.max_model_len - request.num_computed_tokens - self.num_sampled_tokens_per_step)
        if self.need_mamba_block_aligned_split:
            num_new_tokens = self._mamba_block_aligned_split(request, num_new_tokens)
        encoder_inputs_to_schedule = None
        external_load_encoder_input: list[int] = []
        new_encoder_compute_budget = encoder_compute_budget
        if request.has_encoder_inputs:
            encoder_inputs_to_schedule, num_new_tokens, new_encoder_compute_budget, external_load_encoder_input = self._try_schedule_encoder_inputs(request, request.num_computed_tokens, num_new_tokens, encoder_compute_budget, shift_computed_tokens=self.num_prefill_lookahead)
        num_new_tokens = self._reserve_prefill_lookahead(request, request.num_computed_tokens, num_new_tokens)
        if num_new_tokens == 0:
            req_index += 1
            continue
        with record_function_or_nullcontext('schedule: allocate_slots'):
            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is not None:
                    break
                if self.connector is not None and self.connector.has_pending_block_frees():
                    break
                if self.policy == SchedulingPolicy.PRIORITY:
                    preempted_req = self._slai_victim(request, scheduled_new_reqs + scheduled_resumed_reqs + scheduled_running_reqs)
                else:
                    preempted_req = self.running[-1]
                if not self._request_blocks_can_be_freed(preempted_req):
                    break
                if self.policy == SchedulingPolicy.PRIORITY:
                    victim_index = self.running.index(preempted_req)
                    del self.running[victim_index]
                    if victim_index < req_index:
                        req_index -= 1
                    if preempted_req in scheduled_running_reqs:
                        preempted_req_id = preempted_req.request_id
                        scheduled_running_reqs.remove(preempted_req)
                        restored = num_scheduled_tokens.pop(preempted_req_id)
                        token_budget += restored
                        input_budget += restored + draft_slots
                        req_to_new_blocks.pop(preempted_req_id)
                        scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                        preempted_encoder_inputs = scheduled_encoder_inputs.pop(preempted_req_id, None)
                        if preempted_encoder_inputs:
                            num_embeds_to_restore = sum((preempted_req.get_num_encoder_embeds(i) for i in preempted_encoder_inputs))
                            encoder_compute_budget += num_embeds_to_restore
                else:
                    preempted_req = self.running.pop()
                self._preempt_request(preempted_req, scheduled_timestamp, drop_stale_output=self.requires_kv_delivery)
                preempted_reqs.append(preempted_req)
                if preempted_req == request:
                    break
        if new_blocks is None:
            break
        scheduled_running_reqs.append(request)
        prefill_scheduled |= request.is_prefill_chunk
        request_id = request.request_id
        req_to_new_blocks[request_id] = new_blocks
        num_scheduled_tokens[request_id] = num_new_tokens
        token_budget -= num_new_tokens
        input_budget -= num_new_tokens + draft_slots
        req_index += 1
        if request.spec_token_ids:
            num_scheduled_spec_tokens = num_new_tokens + request.num_computed_tokens - request.num_tokens - request.num_output_placeholders
            if num_scheduled_spec_tokens > 0:
                spec_token_ids = request.spec_token_ids
                if len(spec_token_ids) > num_scheduled_spec_tokens:
                    spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                scheduled_spec_decode_tokens[request.request_id] = spec_token_ids
            request.spec_token_ids = []
        if encoder_inputs_to_schedule:
            scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
            for i in encoder_inputs_to_schedule:
                self.encoder_cache_manager.allocate(request, i)
                if self.ec_connector is not None:
                    self.ec_connector.update_state_after_alloc(request, i)
            encoder_compute_budget = new_encoder_compute_budget
        if external_load_encoder_input:
            for i in external_load_encoder_input:
                self.encoder_cache_manager.allocate(request, i)
                if self.ec_connector is not None:
                    self.ec_connector.update_state_after_alloc(request, i)
    total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
    assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
    assert token_budget >= 0
    assert input_budget >= 0
    assert len(self.running) <= self.max_num_running_reqs
    assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(scheduled_running_reqs) <= len(self.running)
    num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
    with record_function_or_nullcontext('schedule: get_num_common_prefix_blocks'):
        if self.running:
            any_request_id = self.running[0].request_id
            num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
    if self.use_v2_model_runner:
        scheduled_new_reqs.extend(scheduled_resumed_reqs)
        scheduled_resumed_reqs.clear()
        new_reqs_data = [NewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids(), req._all_token_ids, uses_mrope=self.model_uses_mrope) for req in scheduled_new_reqs]
    else:
        new_reqs_data = [NewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids(), uses_mrope=self.model_uses_mrope) for req in scheduled_new_reqs]
    with record_function_or_nullcontext('schedule: make_cached_request_data'):
        cached_reqs_data = self._make_cached_request_data(scheduled_running_reqs, scheduled_resumed_reqs, num_scheduled_tokens, scheduled_spec_decode_tokens, req_to_new_blocks)
    if not self.use_v2_model_runner:
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())
    boundary_state_offloads = self.kv_cache_manager.take_boundary_state_offloads()
    kv_connector_block_state = None
    if self.connector is not None:
        block_state_req_ids = set(num_scheduled_tokens)
        block_state_req_ids.update((req_id for req_id in boundary_state_offloads if req_id in self.requests))
        kv_connector_block_state = KVConnectorBlockState(req_ids=block_state_req_ids, resolve_block_ids=self.kv_cache_manager.get_block_ids, boundary_state_offloads=boundary_state_offloads)
    kv_cache_block_copies, cow_retained_blocks = self.kv_cache_manager.take_kv_cache_block_copies()
    if kv_cache_block_copies:
        self._free_cow_retained_blocks(cow_retained_blocks, self.sched_step_seq + 1)
    pending_kv_cache_block_copies = kv_cache_block_copies or None
    num_spec_tokens_to_schedule = self.num_spec_tokens
    if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:
        num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]
    scheduled_encoder_input_stats = None
    if self.log_stats and self.observability_config.enable_logging_iteration_details:
        scheduled_encoder_input_stats = self._make_scheduled_encoder_input_stats(scheduled_encoder_inputs)
    scheduler_output = SchedulerOutput(scheduled_new_reqs=new_reqs_data, scheduled_cached_reqs=cached_reqs_data, num_scheduled_tokens=num_scheduled_tokens, total_num_scheduled_tokens=total_num_scheduled_tokens, scheduled_spec_decode_tokens=scheduled_spec_decode_tokens, scheduled_encoder_inputs=scheduled_encoder_inputs, scheduled_encoder_input_stats=scheduled_encoder_input_stats, num_common_prefix_blocks=num_common_prefix_blocks, preempted_req_ids=self.reset_preempted_req_ids, finished_req_ids=self.finished_req_ids, free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(), new_block_ids_to_zero=self._get_new_block_ids_to_zero(), has_sync_kv_loads=has_sync_kv_loads, kv_cache_block_copies=pending_kv_cache_block_copies, kv_connector_block_state=kv_connector_block_state, num_spec_tokens_to_schedule=num_spec_tokens_to_schedule, ec_manager_metadata=self.encoder_cache_manager.get_manager_metadata())
    if self.connector is not None:
        meta = self._build_kv_connector_meta(self.connector, scheduler_output)
        scheduler_output.kv_connector_metadata = meta
    if self.ec_connector is not None:
        ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(scheduler_output)
        scheduler_output.ec_connector_metadata = ec_meta
    scheduler_output.kv_connector_block_state = None
    if self.defer_block_free and total_num_scheduled_tokens > 0:
        self.sched_step_seq += 1
    with record_function_or_nullcontext('schedule: update_after_schedule'):
        self._update_after_schedule(scheduler_output)
    return scheduler_output
