"""example client implementation for the TRT-LLM served LLM."""

import dataclasses
import random
import time
from typing import List, Sequence

import numpy as np
import tritonclient.grpc as triton_grpcclient
from absl import app, flags, logging
from tritonclient import utils
from tritonclient.grpc import service_pb2, service_pb2_grpc

FLAGS = flags.FLAGS

_SERVER_ADDRESS = flags.DEFINE_string("server_address", "0.0.0.0:8001",
                                      "Address of the text generation server.")


@dataclasses.dataclass(frozen=True)
class InferenceSettings:
    """Settings for the text generation inference."""
    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    stop_word_list: List[str] = None


def _prepare_tensor(name: str, input: np.ndarray) -> triton_grpcclient.InferInput:
    t = triton_grpcclient.InferInput(name, input.shape,
                                     utils.np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


class GrpcTextGenerationServerClient:
    """Client to talk to the text generation server using Triton."""

    def __init__(self, client_stub: triton_grpcclient.InferenceServerClient):
        self._client_stub = client_stub

    def _vectorize_inputs(
            self, prompt: str, inference_settings: InferenceSettings,
            streaming: bool = True
    ) -> List[triton_grpcclient.InferInput]:
        input0_data = np.array([[prompt]]).astype(object)
        output0_len = np.ones_like(input0_data).astype(
            np.uint32) * inference_settings.max_new_tokens
        streaming_data = np.array([[streaming]], dtype=bool)
        stop_words_list = np.array([inference_settings.stop_word_list]).astype(object)
        bad_words = ["shit"]
        bad_words_list = np.array([bad_words]).astype(object)
        tensors = [
            _prepare_tensor("text_input", input0_data),
            _prepare_tensor("max_tokens", output0_len),
            _prepare_tensor("stream", streaming_data),
            _prepare_tensor("pad_id", np.array([[2]], dtype=np.uint32)),
            _prepare_tensor("end_id", np.array([[2]], dtype=np.uint32)),
            _prepare_tensor("top_p",
                            np.array([[inference_settings.top_p]], dtype=np.float32)),
            _prepare_tensor(
                "temperature",
                np.array([[inference_settings.temperature]], dtype=np.float32)),
            _prepare_tensor(
                "repetition_penalty",
                np.array([[inference_settings.repetition_penalty]], dtype=np.float32)),
            _prepare_tensor("stop_words", bad_words_list),
            _prepare_tensor("bad_words", bad_words_list),
        ]

        request = service_pb2.ModelInferRequest(model_name="ensemble",
                                                id=str(random.getrandbits(64)))
        for infer_input in tensors:
            request.inputs.extend([infer_input._get_tensor()])
            if infer_input._get_content() is not None:
                request.raw_input_contents.extend([infer_input._get_content()])
        return request

    def generate(self, prompt: str, inference_settings: InferenceSettings) -> str:
        request = self._vectorize_inputs(prompt, inference_settings, streaming=False)
        response = self._client_stub.ModelInfer(request)
        result = triton_grpcclient.InferResult(response)
        output = result.as_numpy('text_output')
        if output is not None:
            return output[0].decode('utf-8')
        else:
            print("Received an error from server:")
            print(response, result)
            return ""


    def stream_generate(self, prompt: str,
                        inference_settings: InferenceSettings) -> str:
        request = self._vectorize_inputs(prompt, inference_settings)
        text = None
        all_tokens = None

        # Parse the responses
        def get_requests():
            yield request
        for x in self._client_stub.ModelStreamInfer(get_requests()):

            result = triton_grpcclient.InferResult(x.infer_response)
            output = result.as_numpy('text_output')
            if output is not None:
                if text:
                    text += " "
                    text += output.item().decode('utf-8')
                else:
                    text = output.item().decode('utf-8')
                # all_tokens = result.as_numpy('ALL_TOKENS_0').item().decode('utf-8')
            else:
                print("Received an error from server:")
                print(x, result)

        return text, all_tokens


def main(argv: Sequence[str]):
    del argv

    server_address = _SERVER_ADDRESS.value
    logging.info("Connecting to the server at [%s].", server_address)
    channel = triton_grpcclient.grpc.insecure_channel(server_address)
    client_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
    client = GrpcTextGenerationServerClient(client_stub)

    example_prompts = [
        "tell me a joke in one paragraph",
        "The meaning of life is ",
        "If you were to tell me a joke, what would it be?",
    ]
    stop_word_list = ["\n---", "\n--", "\n\n\n", "\n\n#", "\n\nPlease", "\n\nNote:", "##", "XPlayerX:"]
    inference_settings = InferenceSettings(max_new_tokens=120,
                                           temperature=0.7,
                                           top_p=0.9,
                                           repetition_penalty=1.03,
                                           stop_word_list=stop_word_list)

    for prompt in example_prompts:
        logging.info("Prompt: %s", prompt)
        start_time = time.time()
        text, all_tokens = client.stream_generate(prompt, inference_settings)
        logging.info("Generated text: %s", text)
        logging.info("Generated all tokens: %s", all_tokens)
        logging.info("Generation took %s seconds.", time.time() - start_time)


if __name__ == "__main__":
    flags.mark_flag_as_required("server_address")
    app.run(main)