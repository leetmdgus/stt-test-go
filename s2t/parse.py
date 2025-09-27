# # file: run_cpu_transformers.py
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# MODEL_ID = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

# def main(txt_path):
#     # CPU 최적화를 위해 스레드 수 설정 (필요 시 조정)
#     torch.set_num_threads(max(1, torch.get_num_threads()))
#     print(f"PyTorch threads: {torch.get_num_threads()}")

#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_ID,
#         device_map="cpu",
#         torch_dtype=torch.float32,   # CPU는 float32 권장
#         trust_remote_code=True
#     )
#     prompt = "지식 증류(distillation)를 한국어로 쉽게 설명해줘."
#     inputs = tokenizer(prompt, return_tensors="pt")

#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=200,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             repetition_penalty=1.1,
#             eos_token_id=tokenizer.eos_token_id
#         )

#     # 입력 토큰 길이
#     input_length = inputs["input_ids"].shape[1]

#     # 입력 길이 이후만 디코딩
#     generated_tokens = output[0][input_length:]
#     print(tokenizer.decode(generated_tokens, skip_special_tokens=True))


# if __name__ == "__main__":
#     main()
