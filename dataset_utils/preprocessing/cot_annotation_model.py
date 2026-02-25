import os
from vllm import LLM, SamplingParams


class CoTAnnotationModel():
    def __init__(self, config: dict):
        super().__init__()
        model_path = config['pretrained_model_path']
        tensor_parallel_size = config.get('tensor_parallel_size', 1)

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.90,
            max_num_seqs=4,
            limit_mm_per_prompt={"image": 20, "video": 20},
            enforce_eager=True,
            disable_custom_all_reduce=True,
        )
        self.sampling_params = SamplingParams(
            max_tokens=700,
            temperature=0,
        )

    def vlm_inference(self, inputs):
        mm_data = {}
        if inputs.get('image_inputs') is not None:
            mm_data["image"] = inputs['image_inputs']
        if inputs.get('video_inputs') is not None:
            mm_data["video"] = inputs['video_inputs']

        llm_inputs = {
            "prompt": inputs['text'],
            "multi_modal_data": mm_data,
        }

        outputs = self.llm.generate(llm_inputs, sampling_params=self.sampling_params)
        output_text = [output.outputs[0].text for output in outputs]

        return output_text
