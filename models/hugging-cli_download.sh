mkdir chatglm-6b
cd chatglm-6b

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download THUDM/chatglm-6b --local-dir ./

cd ..