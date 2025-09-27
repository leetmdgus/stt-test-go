# file: run_openai_api.py
from openai import OpenAI
import os

def main(txt_path):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = "지식 증류(distillation)를 한국어로 쉽게 설명해줘."

    resp = client.chat.completions.create(
        model="gpt-4o-mini",   # 원하는 모델명 (ex: gpt-3.5-turbo, gpt-4o-mini 등)
        messages=[
            {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7,
        top_p=0.9
    )

    # 답변만 출력 (질문 제외)
    print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()
