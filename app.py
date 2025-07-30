<<<<<<< HEAD
import os
import random
=======
import random
import os
>>>>>>> e27a038 (sync)
import pandas as pd
from pathlib import Path
import gradio as gr
import openai

PROMPT_TEMPLATE = """
Ты выступаешь в роли профессионального макроэкономиста, эксперта по экономике России, денежно-кредитной политике Банка России и финансовым рынкам.
Перед тобой представлены два варианта прогноза ключевых макроэкономических показателей России на конец 2026 и 2027 годов.
Каждый из аналитиков предоставил прогнозные значения в формате JSON, включающие следующие показатели:

CPI (индекс потребительских цен)

GDP (темпы роста ВВП)

KeyRate (ключевая ставка Банка России)

ExchangeRate (курс рубля к доллару США)

UralsPrice (цена нефти марки Urals)

Кроме того, каждый аналитик приложил подробное текстовое обоснование своего прогноза, включающее описание сценариев, предпосылок, макроэкономических условий и причинно-следственных связей.

Твоя задача — определить, какой из двух представленных прогнозов более реалистичен, аргументирован и соответствует текущим и ожидаемым макроэкономическим реалиям России. При принятии решения опирайся на следующие критерии:

реалистичность и обоснованность предпосылок;

логичность причинно-следственных связей в обосновании;

соответствие актуальным и ожидаемым тенденциям экономики России (инфляция, курс рубля, цена на нефть, политика ЦБ РФ).

Для проверки предположений и обоснований, при необходимости, ты можешь использовать интернет для поиска и уточнения актуальной информации по экономике и рынкам. Самостоятельно решай, какие именно данные требуют дополнительной проверки.

Вот два варианта прогнозов и обоснований:

Прогноз аналитика №1:
{model_1}

Прогноз аналитика №2:
{model_2}

После тщательного анализа выбери ТОЛЬКО ОДИН, наиболее реалистичный и аргументированный прогноз.

Ответом должна быть только одна цифра:

1 — если более реалистичен и обоснован прогноз аналитика №1;

2 — если более реалистичен и обоснован прогноз аналитика №2.

Больше никаких комментариев, пояснений или дополнительного текста в ответе быть не должно.
"""


<<<<<<< HEAD

def load_api_key(env_path=".env"):
    """Return API key from .env or environment variable."""
=======
def load_api_key(env_path=".env"):
>>>>>>> e27a038 (sync)
    if os.path.exists(env_path):
        for line in Path(env_path).read_text().splitlines():
            if line.startswith("API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return os.getenv("API_KEY")

def load_agents(csv_path="agents.csv"):
    return pd.read_csv(csv_path)


def get_agent_info(df, agent_id):
    row = df[df["agent_id"] == agent_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def list_agents(answers_dir="answers"):
    return [int(Path(p).stem) for p in Path(answers_dir).glob("*.txt")]


<<<<<<< HEAD
def read_answer(agent_id, answers_dir="answers"):
    return Path(answers_dir) / f"{agent_id}.txt"
=======
def read_answer(agent_id: int, answers_dir: str = "answers") -> str:
    path = Path(answers_dir) / f"{agent_id}.txt"
    return path.read_text(encoding="utf-8")
>>>>>>> e27a038 (sync)


def call_judge(api_key, model_1, model_2, temperature=0.2):
    client = openai.OpenAI(api_key=api_key, base_url="https://bothub.chat/api/v2/openai/v1")
    prompt = PROMPT_TEMPLATE.format(model_1=model_1, model_2=model_2)
<<<<<<< HEAD
=======
    print(prompt)
>>>>>>> e27a038 (sync)
    response = client.chat.completions.create(
        model="o3",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        web_search_options={"search_context_size": "low"},
    )
    content = response.choices[0].message.content.strip()
    return content


def run_tournament(api_key):
    agents_df = load_agents()
    competitors = list_agents()
    random.shuffle(competitors)

    round_num = 1
    result_log = ""
    while len(competitors) > 1:
        result_log += f"\n## Раунд {round_num}\n"
        winners = []
        for i in range(0, len(competitors), 2):
            if i + 1 >= len(competitors):
                winners.append(competitors[i])
                result_log += f"Агент {competitors[i]} проходит без боя.\n"
                continue
            a1 = competitors[i]
            a2 = competitors[i + 1]
<<<<<<< HEAD
            text1 = Path(f"answers/{a1}.txt").read_text()
            text2 = Path(f"answers/{a2}.txt").read_text()
=======
            text1 = read_answer(a1)
            text2 = read_answer(a2)
>>>>>>> e27a038 (sync)
            info1 = get_agent_info(agents_df, a1)
            info2 = get_agent_info(agents_df, a2)
            yield (
                str(info1),
                str(info2),
                text1,
                text2,
                "Сравнение...",
                result_log,
            )
            try:
                choice = call_judge(api_key, text1, text2)
            except Exception as e:
                choice = "error"
                result_log += f"Ошибка сравнения {a1} vs {a2}: {e}\n"
                yield (
                    None,
                    None,
                    None,
                    None,
                    "Ошибка сравнения",
                    result_log,
                )
                continue
            if choice.strip() == "1":
                winner = a1
            else:
                winner = a2
            winners.append(winner)
            result_log += f"Агент {winner} побеждает в паре {a1} vs {a2}.\n"
            yield (
                None,
                None,
                None,
                None,
                f"Победил агент {winner}",
                result_log,
            )
        competitors = winners
        round_num += 1
    champion = competitors[0]
    champion_info = get_agent_info(agents_df, champion)
    result_log += f"\n### Победитель турнира — агент {champion}""\n" + str(champion_info)
    yield (
        "",
        "",
        "",
        "",
        f"Победитель: агент {champion}",
        result_log,
    )


def start(api_key_text):
    key = api_key_text or load_api_key()
<<<<<<< HEAD
    return run_tournament(key)
=======
    for update in run_tournament(key):
        yield update
>>>>>>> e27a038 (sync)


def build_interface():
    with gr.Blocks() as demo:
<<<<<<< HEAD
        gr.Image("logo.png", elem_id="logo")
=======
        gr.Image(
            "logo.png",  # путь к картинке
            elem_id="logo",  # ваш id остаётся
            width=240,  # ширина превью в px
            height="auto",  # можно не задавать, тогда сохранится пропорция
            min_width=240,  # иначе Gradio оставит минимум 160 px
            show_label=False,  # убираем подпись
            show_download_button=False
        )
>>>>>>> e27a038 (sync)
        api_key_inp = gr.Textbox(label="API Key", type="password", value=load_api_key() or "")
        start_btn = gr.Button("Начать соревнование")
        with gr.Row():
            with gr.Column():
                agent1_info = gr.Markdown()
                text1 = gr.Textbox(lines=10, label="Ответ 1")
            with gr.Column():
                agent2_info = gr.Markdown()
                text2 = gr.Textbox(lines=10, label="Ответ 2")
        result = gr.Markdown()
        log = gr.Markdown()

        start_btn.click(
            start,
            inputs=api_key_inp,
            outputs=[
                agent1_info,
                agent2_info,
                text1,
                text2,
                result,
                log,
            ],
        )
    return demo


if __name__ == "__main__":
<<<<<<< HEAD
    build_interface().launch()
=======
    build_interface().launch()
>>>>>>> e27a038 (sync)
