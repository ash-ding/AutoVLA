import os
from vllm import LLM, SamplingParams


class CoTAnnotationModel():
    def __init__(self, config: dict):
        super().__init__()
        model_path = config['pretrained_model_path']
        tensor_parallel_size = config.get('tensor_parallel_size', 1)
        max_num_seqs = config.get('max_num_seqs', config.get('batch_size', 4))

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.90,
            max_num_seqs=max_num_seqs,
            max_model_len=config.get('max_model_len', 8192),
            limit_mm_per_prompt={"image": 20, "video": 20},
            enforce_eager=False,
            disable_custom_all_reduce=True,
        )
        sampling_kwargs = {
            'max_tokens': config.get('max_tokens', 16384),
            'temperature': config.get('temperature', 0),
        }
        # Optional sampling controls — small VL models at greedy decode can
        # fall into repetition loops on long structured prompts; configs can
        # opt in to top_p / top_k / repetition_penalty / seed without breaking
        # callers that don't set them.
        for k in ('top_p', 'top_k', 'repetition_penalty',
                  'frequency_penalty', 'presence_penalty', 'seed'):
            if k in config:
                sampling_kwargs[k] = config[k]
        self.sampling_params = SamplingParams(**sampling_kwargs)

    def _build_llm_input(self, prompt, image_inputs=None, video_inputs=None, mm_processor_kwargs=None):
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        if mm_processor_kwargs:
            llm_inputs["mm_processor_kwargs"] = mm_processor_kwargs

        return llm_inputs

    def vlm_inference(self, inputs):
        if isinstance(inputs.get("text"), list):
            batch_size = len(inputs["text"])
            image_inputs = inputs.get("image_inputs") or [None] * batch_size
            video_inputs = inputs.get("video_inputs") or [None] * batch_size
            mm_processor_kwargs = inputs.get("mm_processor_kwargs") or [{}] * batch_size

            llm_inputs = [
                self._build_llm_input(
                    prompt=inputs["text"][idx],
                    image_inputs=image_inputs[idx],
                    video_inputs=video_inputs[idx],
                    mm_processor_kwargs=mm_processor_kwargs[idx],
                )
                for idx in range(batch_size)
            ]
        else:
            llm_inputs = self._build_llm_input(
                prompt=inputs["text"],
                image_inputs=inputs.get("image_inputs"),
                video_inputs=inputs.get("video_inputs"),
                mm_processor_kwargs=inputs.get("mm_processor_kwargs"),
            )

        outputs = self.llm.generate(
            llm_inputs,
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        output_text = [
            output.outputs[0].text if output.outputs else ""
            for output in outputs
        ]

        return output_text
