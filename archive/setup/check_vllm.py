from vllm import LLM, SamplingParams

if __name__ == "__main__":
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", max_model_len=2048,
              gpu_memory_utilization=0.65, max_num_seqs=4,
              enforce_eager=True, attention_backend="FLASHINFER")
    result = llm.generate(["The capital of France is"],
                          SamplingParams(temperature=0, max_tokens=16))
    assert result[0].outputs[0].token_ids
    print("PASS: vLLM generated tokens using FlashInfer")
    print(result[0].outputs[0].text)
