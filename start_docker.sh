DOCKER_BUILDKIT=1 docker build -t triton_trt_llm -f dockerfile/Dockerfile.trt_llm_backend .

docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
--gpus='"'device=4,5,6,7'"' \
-v /mnt/localdisk/llm-v0/:/models/ \
-v /mnt/localdisk/huggingface:/root/.cache/huggingface/hub \
-v /home/ubuntu/.cache/huggingface/token:/root/.cache/huggingface/token \
-v ${PWD}/triton_model_repo:/triton_model_repo \
-v ${PWD}/scripts/:/scripts  \
-v ${PWD}:/tensorrt_llm_backend \
-w /tensorrt_llm_backend  \
triton_trt_llm /bin/bash
